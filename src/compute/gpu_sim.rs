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

#[derive(Resource, Default, Clone, bevy::render::extract_resource::ExtractResource)]
pub struct ExtractedTextureUpdates {
    pub people: Option<Vec<u8>>,
    pub roads: Option<Vec<u8>>,
    pub road_points: Option<Vec<f32>>,
    pub buildings: Option<Vec<u8>>,
    pub elevations: Option<Vec<f32>>,
}

pub fn apply_texture_updates(
    render_queue: Res<RenderQueue>,
    updates: Res<ExtractedTextureUpdates>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    textures: Res<GpuSimTextures>,
    params: Res<GpuSimParams>,
) {
    if let (Some(data), Some(handle)) = (&updates.people, &textures.people) {
        if let Some(gpu_img) = gpu_images.get(handle) {
            let width = params.people_tex_w;
            let rows = ((data.len() / 16) as u32 + width - 1) / width;
            if rows > 0 {
                render_queue.write_texture(
                    gpu_img.texture.as_image_copy(),
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 16),
                        rows_per_image: None,
                    },
                    Extent3d { width, height: rows, depth_or_array_layers: 1 }
                );
            }
        }
    }
    if let (Some(data), Some(handle)) = (&updates.roads, &textures.roads) {
        if let Some(gpu_img) = gpu_images.get(handle) {
            let width = params.roads_tex_w;
            let rows = ((data.len() / 16) as u32 + width - 1) / width;
            if rows > 0 {
                render_queue.write_texture(
                    gpu_img.texture.as_image_copy(),
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 16),
                        rows_per_image: None,
                    },
                    Extent3d { width, height: rows, depth_or_array_layers: 1 }
                );
            }
        }
    }
    if let (Some(data), Some(handle)) = (&updates.road_points, &textures.road_points) {
        if let Some(gpu_img) = gpu_images.get(handle) {
            let data_bytes: &[u8] = bytemuck::cast_slice(data);
            let width = 1024;
            let rows = ((data_bytes.len() / 8) as u32 + width - 1) / width;
            if rows > 0 {
                render_queue.write_texture(
                    gpu_img.texture.as_image_copy(),
                    data_bytes,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 8), // Rg32Float is 8 bytes
                        rows_per_image: None,
                    },
                    Extent3d { width, height: rows, depth_or_array_layers: 1 }
                );
            }
        }
    }
    if let (Some(data), Some(handle)) = (&updates.buildings, &textures.buildings) {
        if let Some(gpu_img) = gpu_images.get(handle) {
            let width = params.buildings_tex_w;
            let rows = ((data.len() / 16) as u32 + width - 1) / width;
            if rows > 0 {
                render_queue.write_texture(
                    gpu_img.texture.as_image_copy(),
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 16),
                        rows_per_image: None,
                    },
                    Extent3d { width, height: rows, depth_or_array_layers: 1 }
                );
            }
        }
    }
    if let (Some(data), Some(handle)) = (&updates.elevations, &textures.elevations) {
        if let Some(gpu_img) = gpu_images.get(handle) {
            let data_bytes: &[u8] = bytemuck::cast_slice(data);
            let width = params.grid_w;
            let rows = params.grid_h;
            if rows > 0 {
                render_queue.write_texture(
                    gpu_img.texture.as_image_copy(),
                    data_bytes,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4), // R32Float is 4 bytes
                        rows_per_image: None,
                    },
                    Extent3d { width, height: rows, depth_or_array_layers: 1 }
                );
            }
        }
    }
}

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender};

use crate::sim::people::PeopleData;
use crate::sim::buildings::{BuildingData, BUILDING_CAPACITY};
use crate::sim::{ActivityDurations, SimSettings};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable, ShaderType)]
pub struct GpuStats {
    pub people_count: u32,
    pub home_count: u32,
    pub work_count: u32,
    pub shop_count: u32,
    pub travelling_count: u32,
    pub total_money: u32,
    pub residential_occupancy: u32,
    pub office_occupancy: u32,
    pub shop_occupancy: u32,
    pub residential_count: u32,
    pub office_count: u32,
    pub shop_count_b: u32,
    pub residential_assigned: u32,
    pub office_assigned: u32,
    pub _pad: [u32; 2],
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Resource, Default)]
pub struct GpuReadbackBuffer {
    pub inspector_p_buf: Option<Buffer>,
    pub inspector_b_buf: Option<Buffer>,
    pub stats_buffer: Option<Buffer>,
    pub p_mapped: Arc<AtomicBool>,
    pub b_mapped: Arc<AtomicBool>,
    pub s_mapped: Arc<AtomicBool>,
}

#[derive(Resource, Default)]
struct GpuOccupancyBuffer(Option<Buffer>);

fn prepare_readback_buffers(
    render_device: Res<RenderDevice>,
    mut readback: ResMut<GpuReadbackBuffer>,
) {
    if readback.inspector_p_buf.is_none() {
        readback.inspector_p_buf = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("inspector_p_buf"),
            size: 256,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    if readback.inspector_b_buf.is_none() {
        readback.inspector_b_buf = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("inspector_b_buf"),
            size: 256,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    if readback.stats_buffer.is_none() {
        readback.stats_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_stats_readback_buffer"),
            size: std::mem::size_of::<GpuStats>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
}

pub struct GpuSimPlugin;

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct GpuSimLabel;

#[derive(Resource)]
pub struct PeopleReceiver(pub Mutex<Receiver<Vec<(u32, crate::sim::people::PersonRow)>>>);

#[derive(Resource)]
pub struct BuildingsReceiver(pub Mutex<Receiver<Vec<(u32, crate::sim::buildings::BuildingRow)>>>);

#[derive(Resource)]
pub struct PeopleSender(pub Mutex<Sender<Vec<(u32, crate::sim::people::PersonRow)>>>);

#[derive(Resource)]
pub struct BuildingsSender(pub Mutex<Sender<Vec<(u32, crate::sim::buildings::BuildingRow)>>>);

#[derive(Resource)]
pub struct StatsSender(pub Mutex<Sender<GpuStats>>);

#[derive(Resource)]
pub struct StatsReceiver(pub Mutex<Receiver<GpuStats>>);

#[derive(Resource)]
struct GpuSimShader(Handle<Shader>);

#[derive(Resource)]
struct GpuUpdateRoadsShader(Handle<Shader>);

#[derive(Resource)]
struct GpuSpawnShader(Handle<Shader>);

#[derive(Resource, Default)]
struct GpuStatsBuffer(Option<Buffer>);

impl Plugin for GpuSimPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/sim_people.wgsl");
        let update_roads_shader = app.world_mut().resource::<AssetServer>().load("shaders/update_roads.wgsl");
        let spawn_shader = app.world_mut().resource::<AssetServer>().load("shaders/spawn_people.wgsl");
        let (tx_p, rx_p) = std::sync::mpsc::channel();
        let (tx_b, rx_b) = std::sync::mpsc::channel();
        let (tx_s, rx_s) = std::sync::mpsc::channel();
        
        app.insert_resource(PeopleReceiver(Mutex::new(rx_p)));
        app.insert_resource(BuildingsReceiver(Mutex::new(rx_b)));
        app.insert_resource(StatsReceiver(Mutex::new(rx_s)));

        app.add_plugins(ExtractResourcePlugin::<GpuSimParams>::default())
           .add_plugins(ExtractResourcePlugin::<GpuSimTextures>::default())
           .add_plugins(ExtractResourcePlugin::<ExtractedTextureUpdates>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(GpuSimShader(shader))
            .insert_resource(GpuUpdateRoadsShader(update_roads_shader))
            .insert_resource(GpuSpawnShader(spawn_shader))
            .insert_resource(PeopleSender(Mutex::new(tx_p)))
            .insert_resource(BuildingsSender(Mutex::new(tx_b)))
            .insert_resource(StatsSender(Mutex::new(tx_s)))
            .init_resource::<GpuReadbackBuffer>()
            .init_resource::<GpuSimBindGroup>()
            .init_resource::<GpuSimUniformBuffer>()
            .init_resource::<GpuCongestionBuffer>()
            .init_resource::<GpuStatsBuffer>()
            .init_resource::<GpuOccupancyBuffer>()
            .init_resource::<GpuBuildingStatsBuffer>()
            .add_systems(Render, (
                apply_texture_updates,
                prepare_gpu_sim_buffers,
                prepare_readback_buffers,
            ).in_set(bevy::render::RenderSystems::Prepare))
            .add_systems(Render, queue_gpu_sim_bind_group.in_set(bevy::render::RenderSystems::Queue))
            .add_systems(Render, map_and_send_readback.in_set(bevy::render::RenderSystems::Cleanup));

        let mut graph = render_app.world_mut().resource_mut::<bevy::render::render_graph::RenderGraph>();
        graph.add_node(GpuSimLabel, GpuSimNode::default());
        graph.add_node_edge(GpuSimLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GpuSimPipeline>();
    }
}

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct GpuSimTextures {
    pub people: Option<Handle<Image>>,
    pub roads: Option<Handle<Image>>,
    pub buildings: Option<Handle<Image>>,
    pub road_points: Option<Handle<Image>>,
    pub elevations: Option<Handle<Image>>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Resource, ExtractResource, ShaderType)]
pub struct GpuSimParams {
    pub dt: f32,
    pub home_duration: f32,
    pub work_duration: f32,
    pub shop_duration: f32,
    pub home_to_work_prob: f32,
    pub people_count: u32,
    pub people_tex_w: u32,
    pub people_tex_h: u32,
    pub rng_seed: u32,
    pub abandon_multiplier: f32,
    pub rent_cost: f32,
    pub work_salary: f32,
    pub shop_cost: f32,
    pub buildings_tex_w: u32,
    pub buildings_tex_h: u32,
    pub roads_tex_w: u32,
    pub buildings_count: u32,
    pub segments_count: u32,
    pub spawn_count: u32,
    pub spawn_start_index: u32,
    pub b_start: u32,
    pub b_count: u32,
    pub logic_start: u32,
    pub logic_count: u32,
    pub r_start: u32,
    pub r_count: u32,
    pub cycle_frames: u32,
    pub do_stats_readback: u32,
    pub reset_stats: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub entry_seg: u32,
    pub collisions_enabled: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for GpuSimParams {
    fn default() -> Self {
        Self {
            dt: 0.0,
            home_duration: 30.0,
            work_duration: 45.0,
            shop_duration: 15.0,
            home_to_work_prob: 0.6,
            people_count: 0,
            people_tex_w: 0,
            people_tex_h: 0,
            rng_seed: 0,
            abandon_multiplier: 1.0,
            rent_cost: 20.0,
            work_salary: 50.0,
            shop_cost: 30.0,
            buildings_tex_w: 0,
            buildings_tex_h: 0,
            roads_tex_w: 0,
            buildings_count: 0,
            segments_count: 0,
            spawn_count: 0,
            spawn_start_index: 0,
            b_start: 0,
            b_count: 0,
            logic_start: 0,
            logic_count: 0,
            r_start: 0,
            r_count: 0,
            cycle_frames: 90,
            do_stats_readback: 0,
            reset_stats: 0,
            grid_w: 128,
            grid_h: 128,
            entry_seg: 0,
            collisions_enabled: 1.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[derive(Resource)]
struct GpuSimPipeline {
    pub people_pipeline: CachedComputePipelineId,
    pub occupancy_pipeline: CachedComputePipelineId,
    pub logic_pipeline: CachedComputePipelineId,
    pub buildings_pipeline: CachedComputePipelineId,
    pub recount_pipeline: CachedComputePipelineId,
    pub update_roads_pipeline: CachedComputePipelineId,
    pub spawn_pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for GpuSimPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        
        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GpuSimParams::min_size()),
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
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let layout = render_device.create_bind_group_layout(Some("gpu_sim_layout"), &entries);
        let shader = world.resource::<GpuSimShader>().0.clone();
        let pipeline_cache = world.resource::<PipelineCache>();
        
        let layout_desc = BindGroupLayoutDescriptor {
            label: Cow::Borrowed("gpu_sim_layout"),
            entries,
        };

        let people_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_people_movement_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_people_movement")),
            zero_initialize_workgroup_memory: false,
        });
        
        let occupancy_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_occupancy_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_people_occupancy")),
            zero_initialize_workgroup_memory: false,
        });

        let logic_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_people_logic_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_people_logic")),
            zero_initialize_workgroup_memory: false,
        });

        let buildings_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_buildings_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_buildings")),
            zero_initialize_workgroup_memory: false,
        });

        let recount_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_recount_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_recount_stats")),
            zero_initialize_workgroup_memory: false,
        });

        let update_roads_shader = world.resource::<GpuUpdateRoadsShader>().0.clone();
        let update_roads_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_update_roads_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: update_roads_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main")),
            zero_initialize_workgroup_memory: false,
        });

        let spawn_shader = world.resource::<GpuSpawnShader>().0.clone();
        let spawn_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_spawn_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: spawn_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main")),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            people_pipeline,
            occupancy_pipeline,
            buildings_pipeline,
            recount_pipeline,
            update_roads_pipeline,
            logic_pipeline,
            spawn_pipeline,
            bind_group_layout: layout,
        }
    }
}


#[derive(Resource, Default)]
struct GpuSimBindGroup(Option<BindGroup>);

#[derive(Resource, Default)]
struct GpuSimUniformBuffer(Option<Buffer>);

#[derive(Resource, Default)]
struct GpuCongestionBuffer(Option<Buffer>);

#[derive(Resource, Default)]
struct GpuBuildingStatsBuffer(Option<Buffer>);

fn prepare_gpu_sim_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    params: Res<GpuSimParams>,
    _textures: Res<GpuSimTextures>,
    mut buffer: ResMut<GpuSimUniformBuffer>,
    mut congestion: ResMut<GpuCongestionBuffer>,
    mut stats: ResMut<GpuStatsBuffer>,
    mut occupancy: ResMut<GpuOccupancyBuffer>,
    mut b_stats: ResMut<GpuBuildingStatsBuffer>,
) {
    let bytes = bytemuck::bytes_of(&*params);
    if let Some(buf) = &buffer.0 {
        render_queue.write_buffer(buf, 0, bytes);
    } else {
        let b = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gpu_sim_params_buffer"),
            contents: bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        buffer.0 = Some(b);
    }

    // Congestion buffer: segments_count * 4 bytes
    let max_segs = 65536u64; 
    if congestion.0.is_none() || congestion.0.as_ref().unwrap().size() < (max_segs * 4) {
        congestion.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_congestion_buffer"),
            size: max_segs * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    if stats.0.is_none() {
        stats.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_sim_stats_buffer"),
            size: 64, // GpuStats
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    if b_stats.0.is_none() {
        b_stats.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_building_stats_buffer"),
            size: (BUILDING_CAPACITY as u64).max(65536) * 8, // 2 u32 per building (occupants, assigned) = 8 bytes
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    let grid_size = (params.grid_w * params.grid_h) as u64;
    if occupancy.0.is_none() || occupancy.0.as_ref().unwrap().size() < (grid_size * 4) {
        occupancy.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_occupancy_buffer"),
            size: grid_size.max(65536) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
}

fn queue_gpu_sim_bind_group(
    render_device: Res<RenderDevice>,
    pipeline: Res<GpuSimPipeline>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    textures: Res<GpuSimTextures>,
    buffer: Res<GpuSimUniformBuffer>,
    congestion: Res<GpuCongestionBuffer>,
    stats: Res<GpuStatsBuffer>,
    occupancy: Res<GpuOccupancyBuffer>,
    b_stats: Res<GpuBuildingStatsBuffer>,
    path_buffers: Res<crate::compute::gpu_pathfinding::GpuPathBuffers>,
    mut bind_group: ResMut<GpuSimBindGroup>,
) {
    let (Some(people), Some(roads), Some(buildings), Some(params_buf), Some(path_req_buf), Some(paths_buf), Some(congestion_buf), Some(stats_buf), Some(occupancy_buf), Some(b_stats_buf)) = (
        textures.people.as_ref().and_then(|h| gpu_images.get(h)),
        textures.roads.as_ref().and_then(|h| gpu_images.get(h)),
        textures.buildings.as_ref().and_then(|h| gpu_images.get(h)),
        buffer.0.as_ref(),
        path_buffers.requests.as_ref(),
        path_buffers.paths.as_ref(),
        congestion.0.as_ref(),
        stats.0.as_ref(),
        occupancy.0.as_ref(),
        b_stats.0.as_ref(),
    ) else { return; };

    let bg = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            people.texture_view.into_binding(),
            roads.texture_view.into_binding(),
            buildings.texture_view.into_binding(),
            params_buf.as_entire_binding(),
            path_req_buf.as_entire_binding(),
            paths_buf.as_entire_binding(),
            congestion_buf.as_entire_binding(),
            stats_buf.as_entire_binding(),
            occupancy_buf.as_entire_binding(),
            b_stats_buf.as_entire_binding(),
        )),
    );
    bind_group.0 = Some(bg);
}

#[derive(Default)]
struct GpuSimNode;

impl bevy::render::render_graph::Node for GpuSimNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let start = std::time::Instant::now();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_pipeline = world.resource::<GpuSimPipeline>();
        let bind_group = &world.resource::<GpuSimBindGroup>().0;
        let params = world.resource::<GpuSimParams>();
        let textures = world.resource::<GpuSimTextures>();
        let gpu_images = world.resource::<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>();
        let readback = world.resource::<GpuReadbackBuffer>();

        if let (Some(movement_pipe), Some(occupancy_pipe), Some(logic_pipe), Some(build_pipe), Some(recount_pipe), Some(update_roads_pipe), Some(spawn_pipe), Some(bg)) = (
            pipeline_cache.get_compute_pipeline(gpu_pipeline.people_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.occupancy_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.logic_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.buildings_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.recount_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.update_roads_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.spawn_pipeline),
            bind_group
        ) {
            // Clear occupancy buffer every frame
            if let Some(occ_buf) = world.resource::<GpuOccupancyBuffer>().0.as_ref() {
                render_context.command_encoder().clear_buffer(occ_buf, 0, None);
            }

            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_sim_pass"),
                ..default()
            });
            pass.set_bind_group(0, bg, &[]);
            
            // 0. Spawning pass
            if params.spawn_count > 0 {
                pass.set_pipeline(spawn_pipe);
                let spawn_wg_count = (params.spawn_count + 63) / 64;
                pass.dispatch_workgroups(spawn_wg_count, 1, 1);
            }

            // 1. Mark Occupancy pass (to be read by movement pass)
            pass.set_pipeline(occupancy_pipe);
            let p_wg_count = (params.people_count + 63) / 64;
            if p_wg_count > 0 {
                pass.dispatch_workgroups(p_wg_count, 1, 1);
            }

            // End occupancy pass to ensure it's written before movement pass starts
            drop(pass);

            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_sim_movement_pass"),
                ..default()
            });
            pass.set_bind_group(0, bg, &[]);

            // 2. People movement pass (every frame)
            pass.set_pipeline(movement_pipe);
            if p_wg_count > 0 {
                pass.dispatch_workgroups(p_wg_count, 1, 1);
            }
            
            // 3. People logic pass
            if params.logic_count > 0 {
                pass.set_pipeline(logic_pipe);
                let logic_wg_count = (params.logic_count + 63) / 64;
                pass.dispatch_workgroups(logic_wg_count, 1, 1);
            }
            drop(pass);

            // --- RECOUNT PASS ---
            // Clear stats before recount
            if let Some(stats_buf) = world.resource::<GpuStatsBuffer>().0.as_ref() {
                render_context.command_encoder().clear_buffer(stats_buf, 0, None);
            }
            if let Some(b_stats_buf) = world.resource::<GpuBuildingStatsBuffer>().0.as_ref() {
                render_context.command_encoder().clear_buffer(b_stats_buf, 0, None);
            }

            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_sim_recount_pass"),
                ..default()
            });
            pass.set_bind_group(0, bg, &[]);
            
            // 4. Recount pass (1000 people per thread)
            pass.set_pipeline(recount_pipe);
            let num_threads = (params.people_count + 999) / 1000;
            let recount_wg_count = (num_threads + 63) / 64;
            if params.people_count > 0 {
                pass.dispatch_workgroups(recount_wg_count, 1, 1);
            }
            drop(pass);

            // 5. Buildings pass (Updates textures from Recount results)
            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_sim_buildings_pass"),
                ..default()
            });
            pass.set_bind_group(0, bg, &[]);
            if params.b_count > 0 {
                pass.set_pipeline(build_pipe);
                let b_wg_count = (params.b_count + 63) / 64;
                pass.dispatch_workgroups(b_wg_count, 1, 1);
            }

            // 6. Update Roads pass
            if params.r_count > 0 {
                pass.set_pipeline(update_roads_pipe);
                let r_wg_count = (params.r_count + 63) / 64;
                pass.dispatch_workgroups(r_wg_count, 1, 1);
            }
        }

        // Copy Stats
        if params.do_stats_readback > 0 && !readback.s_mapped.load(Ordering::Relaxed) {
            if let (Some(stats_buf), Some(rb_stats_buf)) = (world.resource::<GpuStatsBuffer>().0.as_ref(), readback.stats_buffer.as_ref()) {
                render_context.command_encoder().copy_buffer_to_buffer(stats_buf, 0, rb_stats_buf, 0, 64);
            }
        }

        // Selective Readbacks
        if let Some(selection) = world.get_resource::<crate::ui::inspector::Selection>() {
            match selection.obj {
                Some(crate::ui::inspector::SelectedObj::Person(pid)) => {
                    if !readback.p_mapped.load(Ordering::Relaxed) {
                        if let (Some(people_h), Some(p_buf)) = (textures.people.as_ref(), readback.inspector_p_buf.as_ref()) {
                            if let Some(gpu_img) = gpu_images.get(people_h) {
                                let texel_idx = pid * 3;
                                let x = texel_idx % params.people_tex_w;
                                let y = texel_idx / params.people_tex_w;
                                let mut tex_info = gpu_img.texture.as_image_copy();
                                tex_info.origin = Origin3d { x, y, z: 0 };
                                render_context.command_encoder().copy_texture_to_buffer(
                                    tex_info,
                                    TexelCopyBufferInfo {
                                        buffer: p_buf,
                                        layout: TexelCopyBufferLayout {
                                            offset: 0,
                                            bytes_per_row: Some(256),
                                            rows_per_image: None,
                                        },
                                    },
                                    Extent3d { width: 3, height: 1, depth_or_array_layers: 1 },
                                );
                            }
                        }
                    }
                }
                Some(crate::ui::inspector::SelectedObj::Building(bid)) => {
                    if !readback.b_mapped.load(Ordering::Relaxed) {
                        if let (Some(buildings_h), Some(b_buf)) = (textures.buildings.as_ref(), readback.inspector_b_buf.as_ref()) {
                            if let Some(gpu_img) = gpu_images.get(buildings_h) {
                                let texel_idx = bid * 3;
                                let x = texel_idx % params.buildings_tex_w;
                                let y = texel_idx / params.buildings_tex_w;
                                let mut tex_info = gpu_img.texture.as_image_copy();
                                tex_info.origin = Origin3d { x, y, z: 0 };
                                render_context.command_encoder().copy_texture_to_buffer(
                                    tex_info,
                                    TexelCopyBufferInfo {
                                        buffer: b_buf,
                                        layout: TexelCopyBufferLayout {
                                            offset: 0,
                                            bytes_per_row: Some(256),
                                            rows_per_image: None,
                                        },
                                    },
                                    Extent3d { width: 3, height: 1, depth_or_array_layers: 1 },
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Ok(tx) = world.resource::<crate::TimingsSender>().0.lock() {
            let _ = tx.send((start.elapsed().as_secs_f32() * 1000.0, 0.0));
        }

        Ok(())
    }
}

fn map_and_send_readback(
    _render_device: Res<RenderDevice>,
    readback: Res<GpuReadbackBuffer>,
    sender_p: Res<PeopleSender>,
    sender_b: Res<BuildingsSender>,
    sender_s: Res<StatsSender>,
    params: Res<GpuSimParams>,
    selection: Option<Res<crate::ui::inspector::Selection>>,
) {
    // Stats Readback
    if params.do_stats_readback > 0 && !readback.s_mapped.load(Ordering::Relaxed) {
        if let Some(s_buf) = readback.stats_buffer.as_ref() {
            readback.s_mapped.store(true, Ordering::Relaxed);
            let tx_s = sender_s.0.lock().unwrap().clone();
            let s_clone = s_buf.clone();
            let s_mapped_flag = readback.s_mapped.clone();
            s_buf.slice(..).map_async(MapMode::Read, move |res| {
                if res.is_ok() {
                    let data = s_clone.slice(..).get_mapped_range();
                    let stats: GpuStats = *bytemuck::from_bytes(&data);
                    drop(data);
                    s_clone.unmap();
                    let _ = tx_s.send(stats);
                }
                s_mapped_flag.store(false, Ordering::Relaxed);
            });
        }
    }

    if let Some(sel) = selection {
        match sel.obj {
            Some(crate::ui::inspector::SelectedObj::Person(pid)) => {
                if !readback.p_mapped.load(Ordering::Relaxed) {
                    if let Some(p_buf) = readback.inspector_p_buf.as_ref() {
                        readback.p_mapped.store(true, Ordering::Relaxed);
                        let tx_p = sender_p.0.lock().unwrap().clone();
                        let p_clone = p_buf.clone();
                        let p_mapped_flag = readback.p_mapped.clone();
                        p_buf.slice(..).map_async(MapMode::Read, move |res_p| {
                            if res_p.is_ok() {
                                let data_p = p_clone.slice(..).get_mapped_range();
                                if data_p.len() >= std::mem::size_of::<crate::sim::people::PersonRow>() {
                                    let row: crate::sim::people::PersonRow = *bytemuck::from_bytes(&data_p[..std::mem::size_of::<crate::sim::people::PersonRow>()]);
                                    drop(data_p);
                                    p_clone.unmap();
                                    let _ = tx_p.send(vec![(pid, row)]);
                                } else {
                                    drop(data_p);
                                    p_clone.unmap();
                                }
                            }
                            p_mapped_flag.store(false, Ordering::Relaxed);
                        });
                    }
                }
            }
            Some(crate::ui::inspector::SelectedObj::Building(bid)) => {
                if !readback.b_mapped.load(Ordering::Relaxed) {
                    if let Some(b_buf) = readback.inspector_b_buf.as_ref() {
                        readback.b_mapped.store(true, Ordering::Relaxed);
                        let tx_b = sender_b.0.lock().unwrap().clone();
                        let b_clone = b_buf.clone();
                        let b_mapped_flag = readback.b_mapped.clone();
                        b_buf.slice(..).map_async(MapMode::Read, move |res_b| {
                            if res_b.is_ok() {
                                let data_b = b_clone.slice(..).get_mapped_range();
                                if data_b.len() >= std::mem::size_of::<crate::sim::buildings::BuildingRow>() {
                                    let row: crate::sim::buildings::BuildingRow = *bytemuck::from_bytes(&data_b[..std::mem::size_of::<crate::sim::buildings::BuildingRow>()]);
                                    drop(data_b);
                                    b_clone.unmap();
                                    let _ = tx_b.send(vec![(bid, row)]);
                                } else {
                                    drop(data_b);
                                    b_clone.unmap();
                                }
                            }
                            b_mapped_flag.store(false, Ordering::Relaxed);
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn apply_gpu_readback(
    rx_p: Res<PeopleReceiver>,
    rx_b: Res<BuildingsReceiver>,
    rx_s: Res<StatsReceiver>,
    mut people: ResMut<PeopleData>,
    mut buildings: ResMut<BuildingData>,
    mut grid: ResMut<crate::sim::grid::CityGrid>,
    mut counters: ResMut<crate::sim::counters::SimCounters>,
) {
    if let Ok(rx) = rx_s.0.lock() {
        while let Ok(stats) = rx.try_recv() {
            counters.people = stats.people_count;
            counters.cars = stats.travelling_count;
            counters.res_occupants = stats.residential_occupancy;
            counters.office_occupants = stats.office_occupancy;
            counters.shop_occupants = stats.shop_occupancy;
            counters.residential = stats.residential_count;
            counters.offices = stats.office_count;
            counters.shops = stats.shop_count_b;
            // Optionally update more counters here if needed
        }
    }

    if let Ok(rx) = rx_p.0.lock() {
        while let Ok(p_rows) = rx.try_recv() {
            for (id, row) in p_rows {
                if (id as usize) < people.rows.len() {
                    people.rows[id as usize] = row;
                }
            }
        }
    }
    if let Ok(rx) = rx_b.0.lock() {
        while let Ok(b_rows) = rx.try_recv() {
            for (id, r) in b_rows {
                if let Some(b) = buildings.items.get_mut(id as usize) {
                    b.occupants = r.occupants as u32;
                    b.assigned = r.assigned as u32;
                    b.growth = r.growth;
                    b.age_seconds = r.age_seconds;
                    
                    if b.level != r.level as u32 {
                        b.level = r.level as u32;
                        b.capacity = r.capacity as u32;
                        b.income = r.income;
                    }
                    
                    if b.capacity > 0 && r.capacity <= 0.0 {
                        b.capacity = 0;
                        if let Some(crate::sim::grid::Tile::Building(current_bid)) = grid.get(b.tile.0, b.tile.1) {
                            if current_bid == id {
                                grid.set(b.tile.0, b.tile.1, crate::sim::grid::Tile::Zone(b.btype));
                                counters.destroyed_buildings += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn update_gpu_sim_params(
    time: &Time,
    durations: &ActivityDurations,
    settings: &SimSettings,
    people: &PeopleData,
    buildings: &BuildingData,
    roads: &crate::sim::roads::RoadData,
    grid: &crate::sim::grid::CityGrid,
    gpu_params: &mut GpuSimParams,
    entry_seg: u32,
) {
    gpu_params.dt = time.delta_secs();
    gpu_params.home_duration = durations.home;
    gpu_params.work_duration = durations.work;
    gpu_params.shop_duration = durations.shop;
    gpu_params.home_to_work_prob = durations.home_to_work_prob;
    gpu_params.people_count = people.len;
    gpu_params.people_tex_w = people.tex_width;
    gpu_params.people_tex_h = people.tex_height;
    gpu_params.rng_seed = rand::random();
    gpu_params.abandon_multiplier = settings.abandon_multiplier;
    gpu_params.rent_cost = settings.rent_cost;
    gpu_params.work_salary = settings.work_salary;
    gpu_params.shop_cost = settings.shop_cost;
    gpu_params.buildings_tex_w = buildings.tex_width;
    gpu_params.buildings_tex_h = buildings.tex_height;
    gpu_params.roads_tex_w = roads.tex_width;
    gpu_params.buildings_count = buildings.items.len() as u32;
    gpu_params.segments_count = roads.segments.len() as u32;
    gpu_params.grid_w = grid.width;
    gpu_params.grid_h = grid.height;
    gpu_params.entry_seg = entry_seg;
    gpu_params.collisions_enabled = settings.collisions_enabled;
}
