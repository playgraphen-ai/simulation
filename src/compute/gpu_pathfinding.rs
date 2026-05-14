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
    pub _pad: u32,
}

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct PathRequestsResource {
    pub list: Vec<PathRequest>,
}

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct ExtractedMajorGraph {
    pub rows: Vec<crate::sim::roads::MajorRoadRow>,
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
    pub major_segments_count: u32,
}

impl Default for PathParams {
    fn default() -> Self {
        Self { 
            segments_count: 0, 
            roads_tex_w: 0, 
            max_path_len: 64, 
            request_count: 0, 
            slice_start: 0, 
            slice_end: 0, 
            do_dispatch: 0, 
            reset_path_queue: 0,
            major_segments_count: 0,
        }
    }
}

#[derive(Resource)]
struct GpuPathfindingShader(Handle<Shader>);

impl Plugin for GpuPathfindingPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/pathfind.wgsl");
        app.init_resource::<PathParams>();
        app.init_resource::<PathRequestsResource>();
        app.init_resource::<ExtractedMajorGraph>();

        app.add_plugins(ExtractResourcePlugin::<PathParams>::default());
        app.add_plugins(ExtractResourcePlugin::<PathRequestsResource>::default());
        app.add_plugins(ExtractResourcePlugin::<ExtractedMajorGraph>::default());
        
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
    pub major_graph: Option<Buffer>,
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
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
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
    major_graph: Res<ExtractedMajorGraph>,
    mut buffers: ResMut<GpuPathBuffers>,
) {
    let max_reqs = 65536u32;
    buffers.max_requests = max_reqs;

    // IMPORTANT: Sync max_path_len to match the search window.
    params.max_path_len = 512;
    params.major_segments_count = major_graph.rows.len() as u32;

    let people_capacity = 65536u64; // Max people
    let paths_size = people_capacity * 512u64 * 4u64; // 512 max path len * 4 bytes per id
    if buffers.paths.is_none() {
        let initial_data = vec![0xFFu8; paths_size as usize];
        buffers.paths = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("path_results_buffer"),
            contents: &initial_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::MAP_READ | BufferUsages::COPY_DST, 
        }));
    }

    if buffers.requests.is_none() {
        buffers.requests = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("path_requests_buffer"),
            size: 16 + (max_reqs as u64 * 16), // 16 bytes per PathRequest
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::MAP_READ | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
    }

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

    // Fixed size: 65536 requests * 512 entries * 4 bytes per entry (u16 key, u16 val) = 128MB
    let prev_size = max_reqs as u64 * 512u64 * 4u64;
    if buffers.prev.is_none() || buffers.prev.as_ref().unwrap().size() != prev_size {
        buffers.prev = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("path_prev_buffer"),
            size: prev_size, 
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }

    // Major Graph buffer
    if !major_graph.rows.is_empty() {
        let size = (major_graph.rows.len() * std::mem::size_of::<crate::sim::roads::MajorRoadRow>()) as u64;
        if buffers.major_graph.is_none() || buffers.major_graph.as_ref().unwrap().size() < size {
            buffers.major_graph = Some(render_device.create_buffer(&BufferDescriptor {
                label: Some("major_graph_buffer"),
                size: size.max(1024), // Minimum size
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if let Some(buf) = &buffers.major_graph {
            render_queue.write_buffer(buf, 0, bytemuck::cast_slice(&major_graph.rows));
        }
    } else if buffers.major_graph.is_none() {
        // Create a dummy buffer if empty
        buffers.major_graph = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("major_graph_dummy"),
            size: 1024,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            buffers.major_graph.as_ref().unwrap().as_entire_binding(),
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
        let start = std::time::Instant::now();
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
            // Let the GPU decide the number of workgroups based on the indirect buffer count_x
            pass.dispatch_workgroups_indirect(buffers.requests.as_ref().unwrap(), 0);
        }

        if let Ok(tx) = world.resource::<crate::TimingsSender>().0.lock() {
            let mut event = crate::TimingEvent::default();
            event.pathfind = start.elapsed().as_secs_f32() * 1000.0;
            let _ = tx.send(event);
        }

        Ok(())
    }
}
