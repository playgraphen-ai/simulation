use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{RenderLabel, NodeRunError, RenderGraphContext},
    },
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use crate::compute::gpu_sim::GpuSimTextures;

pub struct GpuPickingPlugin;

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct GpuPickingLabel;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType, ExtractResource, Resource)]
pub struct PickingParams {
    pub ray_origin: Vec4,
    pub ray_dir: Vec4,
    pub is_active: u32,
    pub max_cars: u32,
    pub transforms_w: u32,
    pub pad: u32,
}

#[derive(Resource)]
struct GpuPickingShader(Handle<Shader>);

impl Plugin for GpuPickingPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/picking.wgsl");
        app.init_resource::<PickingParams>();
        app.init_resource::<PickedCar>();
        
        let mut buffers = app.world_mut().resource_mut::<Assets<bevy::render::storage::ShaderStorageBuffer>>();
        let mut buffer = bevy::render::storage::ShaderStorageBuffer::from(vec![0u32; 2]);
        buffer.buffer_description.usage |= bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::COPY_SRC;
        let handle = buffers.add(buffer);
        app.insert_resource(PickingResultBufferHandle(handle.clone()));
        
        app.add_plugins(ExtractResourcePlugin::<PickingParams>::default());
        app.add_plugins(ExtractResourcePlugin::<PickingResultBufferHandle>::default());

        app.add_systems(Update, (update_picking_ray, request_picking_readback));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(GpuPickingShader(shader))
            .init_resource::<GpuPickingBuffers>()
            .add_systems(Render, prepare_picking_buffers.in_set(bevy::render::RenderSystems::Prepare))
            .add_systems(Render, queue_picking_bind_group.in_set(bevy::render::RenderSystems::Queue));

        let mut graph = render_app.world_mut().resource_mut::<bevy::render::render_graph::RenderGraph>();
        graph.add_node(GpuPickingLabel, GpuPickingNode::default());
        graph.add_node_edge(crate::compute::gpu_sim::GpuSimLabel, GpuPickingLabel);
        graph.add_node_edge(GpuPickingLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GpuPickingPipeline>();
    }
}

fn request_picking_readback(
    mut commands: Commands,
    params: Res<PickingParams>,
    picking_res: Option<Res<PickingResultBufferHandle>>,
    mut last_active: Local<u32>,
) {
    if params.is_active > 0 && *last_active == 0 {
        if let Some(res_buf) = picking_res {
            commands.spawn(bevy::render::gpu_readback::Readback::buffer(res_buf.0.clone()))
                .observe(|trigger: bevy::ecs::observer::On<bevy::render::gpu_readback::ReadbackComplete>, mut picked: ResMut<PickedCar>| {
                    let data = trigger.event().to_shader_type::<[u32; 2]>();
                    let id = data[0];
                    if id != 0xFFFFFFFF {
                        picked.0 = Some(id);
                    } else {
                        picked.0 = None;
                    }
                });
        }
    }
    *last_active = params.is_active;
}

#[derive(Resource, Clone, ExtractResource)]
pub struct PickingResultBufferHandle(pub Handle<bevy::render::storage::ShaderStorageBuffer>);


#[derive(Resource, Default)]
pub struct PickedCar(pub Option<u32>);

fn update_picking_ray(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut params: ResMut<PickingParams>,
    people: Res<crate::sim::people::PeopleData>,
    active_tool: Res<crate::ui::tools::ActiveTool>,
) {
    params.is_active = 0;
    if !mouse.just_pressed(MouseButton::Left) { return; }
    if !matches!(*active_tool, crate::ui::tools::ActiveTool::None) { return; }

    if let Ok(window) = windows.single() {
        if let Some(pos) = window.cursor_position() {
            if let Ok((camera, cam_tf)) = camera_q.single() {
                if let Ok(ray) = camera.viewport_to_world(cam_tf, pos) {
                    params.ray_origin = ray.origin.extend(1.0);
                    params.ray_dir = Vec3::from(ray.direction).extend(0.0);
                    params.is_active = 1;
                    params.max_cars = people.len;
                    params.transforms_w = 1024;
                }
            }
        }
    }
}

#[derive(Resource)]
struct GpuPickingPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for GpuPickingPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(PickingParams::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let layout = render_device.create_bind_group_layout(Some("gpu_picking_layout"), &entries);
        let shader = world.resource::<GpuPickingShader>().0.clone();
        let pipeline_cache = world.resource::<PipelineCache>();
        
        let layout_desc = BindGroupLayoutDescriptor {
            label: Cow::Borrowed("gpu_picking_layout"),
            entries,
        };

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_picking_pipeline")),
            layout: vec![layout_desc],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main")),
            zero_initialize_workgroup_memory: false,
        });

        Self { pipeline, bind_group_layout: layout }
    }
}

#[derive(Resource, Default)]
struct GpuPickingBuffers {
    pub params: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
}

fn prepare_picking_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut buffers: ResMut<GpuPickingBuffers>,
    params: Res<PickingParams>,
) {
    let param_bytes = bytemuck::bytes_of(&*params);
    if let Some(buf) = &buffers.params {
        render_queue.write_buffer(buf, 0, param_bytes);
    } else {
        buffers.params = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gpu_picking_params_buffer"),
            contents: param_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    }
}

fn queue_picking_bind_group(
    render_device: Res<RenderDevice>,
    pipeline: Res<GpuPickingPipeline>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    textures: Res<GpuSimTextures>,
    mut buffers: ResMut<GpuPickingBuffers>,
    picking_res: Option<Res<PickingResultBufferHandle>>,
    storage_buffers: Res<bevy::render::render_asset::RenderAssets<bevy::render::storage::GpuShaderStorageBuffer>>,
) {
    let (Some(transforms), Some(params_buf)) = (
        textures.car_transforms.as_ref().and_then(|h| gpu_images.get(h)),
        buffers.params.as_ref(),
    ) else { return; };
    
    let res_buf_binding = if let Some(handle) = picking_res {
        if let Some(gpu_buf) = storage_buffers.get(&handle.0) {
            Some(gpu_buf.buffer.as_entire_binding())
        } else {
            None
        }
    } else {
        None
    };
    
    let Some(res_buf) = res_buf_binding else { return; };

    let bg = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            transforms.texture_view.into_binding(),
            params_buf.as_entire_binding(),
            res_buf,
        )),
    );
    buffers.bind_group = Some(bg);
}

#[derive(Default)]
struct GpuPickingNode;

impl bevy::render::render_graph::Node for GpuPickingNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let params = world.resource::<PickingParams>();
        if params.is_active == 0 {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_id = world.resource::<GpuPickingPipeline>().pipeline;
        let buffers = world.resource::<GpuPickingBuffers>();

        let res_buf_binding = if let Some(handle) = world.get_resource::<PickingResultBufferHandle>() {
            let storage_buffers = world.resource::<bevy::render::render_asset::RenderAssets<bevy::render::storage::GpuShaderStorageBuffer>>();
            if let Some(gpu_buf) = storage_buffers.get(&handle.0) {
                Some(gpu_buf.buffer.clone())
            } else {
                None
            }
        } else {
            None
        };

        if let (Some(pipeline), Some(bg), Some(result)) = (
            pipeline_cache.get_compute_pipeline(pipeline_id),
            buffers.bind_group.as_ref(),
            res_buf_binding,
        ) {
            render_context.command_encoder().clear_buffer(&result, 0, Some(8));

            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_picking_pass"),
                ..default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            let wg_count = (params.max_cars + 63) / 64;
            if wg_count > 0 {
                pass.dispatch_workgroups(wg_count, 1, 1);
            }
            drop(pass);
        }

        Ok(())
    }
}

