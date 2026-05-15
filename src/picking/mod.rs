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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

pub fn process_pick_result(
    receiver: Res<PickReceiver>,
    mut picked: ResMut<PickedCar>,
) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok(id) = rx.try_recv() {
            picked.0 = id;
        }
    }
}

impl Plugin for GpuPickingPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/picking.wgsl");
        app.init_resource::<PickingParams>();
        app.init_resource::<PickedCar>();
        
        let (tx, rx) = std::sync::mpsc::channel();
        app.insert_resource(PickSender(std::sync::Mutex::new(tx.clone())));
        app.insert_resource(PickReceiver(std::sync::Mutex::new(rx)));
        
        app.add_plugins(ExtractResourcePlugin::<PickingParams>::default());

        app.add_systems(Update, (update_picking_ray, process_pick_result));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(GpuPickingShader(shader))
            .insert_resource(PickSender(std::sync::Mutex::new(tx)))
            .init_resource::<GpuPickingBuffers>()
            .add_systems(Render, prepare_picking_buffers.in_set(bevy::render::RenderSystems::Prepare))
            .add_systems(Render, queue_picking_bind_group.in_set(bevy::render::RenderSystems::Queue))
            .add_systems(Render, readback_picking_result.in_set(bevy::render::RenderSystems::Cleanup));

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
    pub result: Option<Buffer>,
    pub readback: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
    pub readback_mapped: Arc<AtomicBool>,
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

    if buffers.result.is_none() {
        buffers.result = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_picking_result_buffer"),
            size: 8, // u32 id, f32 dist
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    if buffers.readback.is_none() {
        buffers.readback = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_picking_readback_buffer"),
            size: 8,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
}

fn queue_picking_bind_group(
    render_device: Res<RenderDevice>,
    pipeline: Res<GpuPickingPipeline>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    textures: Res<GpuSimTextures>,
    mut buffers: ResMut<GpuPickingBuffers>,
) {
    let (Some(transforms), Some(params_buf), Some(result_buf)) = (
        textures.car_transforms.as_ref().and_then(|h| gpu_images.get(h)),
        buffers.params.as_ref(),
        buffers.result.as_ref(),
    ) else { return; };

    let bg = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            transforms.texture_view.into_binding(),
            params_buf.as_entire_binding(),
            result_buf.as_entire_binding(),
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

        if let (Some(pipeline), Some(bg), Some(result), Some(readback)) = (
            pipeline_cache.get_compute_pipeline(pipeline_id),
            buffers.bind_group.as_ref(),
            buffers.result.as_ref(),
            buffers.readback.as_ref()
        ) {
            // Clear result buffer: ID = 0xFFFFFFFF, Dist = INF (f32::MAX bit pattern)
            let _clear_data: [u32; 2] = [0xFFFFFFFF, 0x7F7FFFFF];
            render_context.command_encoder().clear_buffer(result, 0, Some(8));
            // Actually, clear buffer only writes zeros.
            // Let's copy a small initialization buffer instead.
            // But we don't have one prepared. Let's do it in the shader on ID 0.

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

            render_context.command_encoder().copy_buffer_to_buffer(result, 0, readback, 0, 8);
        }

        Ok(())
    }
}

use std::sync::mpsc::{Sender, Receiver};

#[derive(Resource)]
pub struct PickSender(pub std::sync::Mutex<Sender<Option<u32>>>);

#[derive(Resource)]
pub struct PickReceiver(pub std::sync::Mutex<Receiver<Option<u32>>>);

fn readback_picking_result(
    _render_device: Res<RenderDevice>,
    buffers: ResMut<GpuPickingBuffers>,
    params: Res<PickingParams>,
    sender: Res<PickSender>,
) {
    if params.is_active > 0 && !buffers.readback_mapped.load(Ordering::Relaxed) {
        if let Some(readback) = buffers.readback.as_ref() {
            buffers.readback_mapped.store(true, Ordering::Relaxed);
            
            let r_clone = readback.clone();
            let mapped_flag = buffers.readback_mapped.clone();
            let tx = sender.0.lock().unwrap().clone();
            
            readback.slice(..).map_async(MapMode::Read, move |res| {
                if res.is_ok() {
                    let data = r_clone.slice(..).get_mapped_range();
                    let res_data: [u32; 2] = *bytemuck::from_bytes(&data[0..8]);
                    drop(data);
                    r_clone.unmap();
                    
                    let id = res_data[0];
                    if id != 0xFFFFFFFF {
                        let _ = tx.send(Some(id));
                    } else {
                        let _ = tx.send(None);
                    }
                }
                mapped_flag.store(false, Ordering::Relaxed);
            });
        }
    }
}
