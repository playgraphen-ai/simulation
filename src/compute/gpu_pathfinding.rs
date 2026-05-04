use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderContext, RenderQueue},
        Render, RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{RenderLabel, NodeRunError, RenderGraphContext},
    },
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

pub struct GpuPathfindingPlugin;

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct GpuPathfindingLabel;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PathRequest {
    pub start: u32,
    pub target_seg: u32,
    pub person_id: u32,
}

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct PathRequestsResource {
    pub list: Vec<PathRequest>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Resource, ExtractResource, ShaderType)]
pub struct PathParams {
    pub segments_count: u32,
    pub roads_tex_w: u32,
    pub max_path_len: u32,
    pub request_count: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub do_dispatch: u32,
    pub reset_path_queue: u32,
}

impl Default for PathParams {
    fn default() -> Self {
        Self { segments_count: 0, roads_tex_w: 0, max_path_len: 64, request_count: 0, slice_start: 0, slice_end: 0, do_dispatch: 0, reset_path_queue: 0 }
    }
}

#[derive(Resource)]
struct GpuPathfindingShader(Handle<Shader>);

impl Plugin for GpuPathfindingPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/pathfind.wgsl");
        app.init_resource::<PathParams>();
        app.init_resource::<PathRequestsResource>();

        app.add_plugins(ExtractResourcePlugin::<PathParams>::default());
        app.add_plugins(ExtractResourcePlugin::<PathRequestsResource>::default());
        
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(GpuPathfindingShader(shader))
            .init_resource::<GpuPathBuffers>()
            .add_systems(Render, prepare_path_buffers.in_set(bevy::render::RenderSystems::Prepare));

        let mut graph = render_app.world_mut().resource_mut::<bevy::render::render_graph::RenderGraph>();
        graph.add_node(GpuPathfindingLabel, GpuPathfindingNode::default());
        graph.add_node_edge(crate::compute::gpu_sim::GpuSimLabel, GpuPathfindingLabel);
        graph.add_node_edge(GpuPathfindingLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GpuPathfindingPipeline>();
    }
}

#[derive(Resource, Default)]
pub struct GpuPathBuffers {
    pub requests: Option<Buffer>,
    pub prev: Option<Buffer>,
    pub paths: Option<Buffer>,
    pub params: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
    pub max_requests: u32,
}

#[derive(Resource)]
struct GpuPathfindingPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for GpuPathfindingPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
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
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(PathParams::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let layout = render_device.create_bind_group_layout(Some("gpu_path_layout"), &entries);
        let shader = world.resource::<GpuPathfindingShader>().0.clone();
        let pipeline_cache = world.resource::<PipelineCache>();
        
        let layout_desc = BindGroupLayoutDescriptor {
            label: Cow::Borrowed("gpu_sim_layout"),
            entries,
        };

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_path_pipeline")),
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

fn prepare_path_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<GpuPathfindingPipeline>,
    mut params: ResMut<PathParams>,
    requests: Res<PathRequestsResource>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    gpu_sim_textures: Res<crate::compute::gpu_sim::GpuSimTextures>,
    mut buffers: ResMut<GpuPathBuffers>,
) {
    let max_reqs = 16384u32;
    buffers.max_requests = max_reqs;

    // IMPORTANT: Sync max_path_len to match shader's 256.
    params.max_path_len = 256;

    // We can't access `roads: Res<RoadData>` here easily without moving it to RenderApp or syncing it.
    // Wait, GpuSimParams has `roads_tex_w`, we can read from there or pass it.
    // Actually, `params.roads_tex_w` and `params.segments_count` are currently never updated in the RenderApp!
    // We must extract them or update them.


    if buffers.requests.is_none() {
        buffers.requests = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("path_requests_buffer"),
            size: 16 + (max_reqs as u64 * 12),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::MAP_READ | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
    }

    // Mix CPU requests with GPU indirect buffer
    let cpu_req_count = requests.list.len().min(max_reqs as usize) as u32;
    if let Some(req_buf) = &buffers.requests {
        if params.reset_path_queue == 1 {
            let header = [cpu_req_count, 1u32, 1u32, 0u32];
            render_queue.write_buffer(req_buf, 0, bytemuck::cast_slice(&header));
            if cpu_req_count > 0 {
                render_queue.write_buffer(req_buf, 16, bytemuck::cast_slice(&requests.list[..cpu_req_count as usize]));
            }
        }
    }

    // Max buffer size for prev array is limit to 512MB
    let max_prev_size = 67108864u64; // 64MB
    let prev_size = (max_reqs as u64 * params.segments_count as u64 * 4).max(4).min(max_prev_size);
    if buffers.prev.is_none() || buffers.prev.as_ref().unwrap().size() < prev_size {
        buffers.prev = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("path_prev_buffer"),
            size: max_prev_size, // Use fixed max allowed size
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }

    let people_capacity = 65536u64; // Max people
    let paths_size = people_capacity * 256u64 * 4u64; // 256 max path len * 4 bytes per id
    if buffers.paths.is_none() || buffers.paths.as_ref().unwrap().size() < paths_size {
        // Try to initialize it with clear_buffer if possible, or initialize with zero and rely on shader, but shader requires 0xFF.
        // Let's create it mapped and fill it, or use command encoder.
        // For simplicity, we will just allocate the 64MB vector.
        let initial_data = vec![0xFFu8; paths_size as usize];
        buffers.paths = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("path_results_buffer"),
            contents: &initial_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::MAP_READ | BufferUsages::COPY_DST, 
        }));
    }

    let param_bytes = bytemuck::bytes_of(&*params);
    if let Some(buf) = &buffers.params {
        render_queue.write_buffer(buf, 0, param_bytes);
    } else {
        buffers.params = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("path_params_buffer"),
            contents: param_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    }

    let Some(roads_gpu) = gpu_sim_textures.roads.as_ref().and_then(|h| gpu_images.get(h)) else { return; };

    let bg = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            roads_gpu.texture_view.into_binding(),
            buffers.prev.as_ref().unwrap().as_entire_binding(),
            buffers.paths.as_ref().unwrap().as_entire_binding(),
            buffers.params.as_ref().unwrap().as_entire_binding(),
            buffers.requests.as_ref().unwrap().as_entire_binding(),
        )),
    );
    buffers.bind_group = Some(bg);
}

#[derive(Default)]
struct GpuPathfindingNode;

impl bevy::render::render_graph::Node for GpuPathfindingNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let params = world.resource::<PathParams>();
        if params.do_dispatch == 0 {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_id = world.resource::<GpuPathfindingPipeline>().pipeline;
        let buffers = world.resource::<GpuPathBuffers>();

        if let (Some(pipeline), Some(bg)) = (pipeline_cache.get_compute_pipeline(pipeline_id), buffers.bind_group.as_ref()) {
            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_pathfinding_pass"),
                ..default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            // Dispatch 274 workgroups per frame (16384 max paths / 60 frames = 273.06)
            pass.dispatch_workgroups(274, 1, 1);
        }
        Ok(())
    }
}
