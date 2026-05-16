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

use crate::sim::people::PeopleData;
use crate::sim::buildings::BuildingData;
use crate::sim::{SimConfig, SimSettings};

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
    pub bankrupt_count: u32,
    pub active_car_count: u32, // New field, cleared every frame
    pub tax_income_total: u32,
    pub tax_rent_total: u32,
    pub tax_consumption_total: u32,
    pub live_car_count: u32,
}

use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Resource, Default)]
struct GpuOccupancyBuffer(Option<Buffer>);

pub struct GpuSimPlugin;

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct GpuSimLabel;

#[derive(Resource)]
struct GpuSimShader(Handle<Shader>);

#[derive(Resource)]
struct GpuUpdateRoadsShader(Handle<Shader>);

#[derive(Resource)]
struct GpuSpawnShader(Handle<Shader>);

#[derive(Resource, Default)]
struct GpuStatsBuffer(Option<Buffer>);

#[derive(Resource, Clone, ExtractResource)]
pub struct GpuReadbackBufferHandles {
    pub inspector_p_buf: Handle<bevy::render::storage::ShaderStorageBuffer>,
    pub inspector_b_buf: Handle<bevy::render::storage::ShaderStorageBuffer>,
    pub stats_buffer: Handle<bevy::render::storage::ShaderStorageBuffer>,
}

impl Plugin for GpuSimPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/sim_people.wgsl");
        let update_roads_shader = app.world_mut().resource::<AssetServer>().load("shaders/update_roads.wgsl");
        let spawn_shader = app.world_mut().resource::<AssetServer>().load("shaders/spawn_people.wgsl");
        let car_transform_shader = app.world_mut().resource::<AssetServer>().load("shaders/car_transform.wgsl");

        app.add_systems(Update, request_gpu_readback);

        let mut buffers = app.world_mut().resource_mut::<Assets<bevy::render::storage::ShaderStorageBuffer>>();
        
        let mut p_buf = bevy::render::storage::ShaderStorageBuffer::from(vec![0u32; 64]);
        p_buf.buffer_description.usage |= bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::COPY_SRC;
        
        let mut b_buf = bevy::render::storage::ShaderStorageBuffer::from(vec![0u32; 64]);
        b_buf.buffer_description.usage |= bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::COPY_SRC;
        
        let mut s_buf = bevy::render::storage::ShaderStorageBuffer::from(vec![0u32; std::mem::size_of::<GpuStats>() / 4 + 1]);
        s_buf.buffer_description.usage |= bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::COPY_SRC;

        let handles = GpuReadbackBufferHandles {
            inspector_p_buf: buffers.add(p_buf),
            inspector_b_buf: buffers.add(b_buf),
            stats_buffer: buffers.add(s_buf),
        };
        app.insert_resource(handles);

        app.add_plugins(ExtractResourcePlugin::<GpuSimParams>::default())
           .add_plugins(ExtractResourcePlugin::<GpuSimTextures>::default())
           .add_plugins(ExtractResourcePlugin::<ExtractedTextureUpdates>::default())
           .add_plugins(ExtractResourcePlugin::<GpuReadbackBufferHandles>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(GpuSimShader(shader))
            .insert_resource(GpuUpdateRoadsShader(update_roads_shader))
            .insert_resource(GpuSpawnShader(spawn_shader))
            .insert_resource(GpuCarTransformShader(car_transform_shader))
            .init_resource::<GpuSimBindGroup>()
            .init_resource::<GpuSimUniformBuffer>()
            .init_resource::<GpuCongestionBuffer>()
            .init_resource::<GpuStatsBuffer>()
            .init_resource::<GpuOccupancyBuffer>()
            .init_resource::<GpuBuildingStatsBuffer>()
            .init_resource::<GpuLastCopyFrame>()
            .add_systems(Render, (
                apply_texture_updates,
                prepare_gpu_sim_buffers,
            ).in_set(bevy::render::RenderSystems::Prepare))
            .add_systems(Render, queue_gpu_sim_bind_group.in_set(bevy::render::RenderSystems::Queue));

        let mut graph = render_app.world_mut().resource_mut::<bevy::render::render_graph::RenderGraph>();
        graph.add_node(GpuSimLabel, GpuSimNode::default());
        graph.add_node_edge(GpuSimLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GpuSimPipeline>();
    }
}

#[derive(Resource, Default)]
struct GpuLastCopyFrame(AtomicU32);

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct GpuSimTextures {
    pub people: Option<Handle<Image>>,
    pub roads: Option<Handle<Image>>,
    pub buildings: Option<Handle<Image>>,
    pub road_points: Option<Handle<Image>>,
    pub elevations: Option<Handle<Image>>,
    pub car_transforms: Option<Handle<Image>>,
}

use crate::sim::constants::{MAX_PEOPLE, MAX_BUILDINGS, MAX_SEGMENTS, MIN_GPU_BUFFER_CAPACITY, MAX_PATH_LEN, MAX_PATH_REQUESTS};

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
    pub tax_income: f32,
    pub tax_rent: f32,
    pub tax_consumption: f32,
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
    pub recount_slice: u32,
    pub recount_slice_count: u32,
    pub do_occupancy_gc: u32,
    pub do_inspector_readback: u32,
    pub car_capacity: u32,
    pub max_people: u32,
    pub max_segments: u32,
    pub max_buildings: u32,
    pub max_path_len: u32,
    pub max_path_requests: u32,
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
            tax_income: 0.15,
            tax_rent: 0.1,
            tax_consumption: 0.08,
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
            recount_slice: 0,
            recount_slice_count: 5,
            do_occupancy_gc: 0,
            do_inspector_readback: 0,
            car_capacity: 0,
            max_people: MAX_PEOPLE,
            max_segments: MAX_SEGMENTS,
            max_buildings: MAX_BUILDINGS,
            max_path_len: MAX_PATH_LEN,
            max_path_requests: MAX_PATH_REQUESTS,
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
    pub recalibrate_pipeline: CachedComputePipelineId,
    pub occupancy_gc_clear_pipeline: CachedComputePipelineId,
    pub occupancy_gc_repopulate_pipeline: CachedComputePipelineId,
    pub update_roads_pipeline: CachedComputePipelineId,
    pub spawn_pipeline: CachedComputePipelineId,
    pub car_transform_pipeline: CachedComputePipelineId,
    pub car_clear_pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

#[derive(Resource)]
struct GpuCarTransformShader(Handle<Shader>);

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
            // NEW BINDINGS FOR CAR TRANSFORMS
            BindGroupLayoutEntry {
                binding: 10,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 11,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 12,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 13,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
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

        let recalibrate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_recalibrate_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_recalibrate_stats")),
            zero_initialize_workgroup_memory: false,
        });

        let occupancy_gc_clear_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_occupancy_gc_clear_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_occupancy_gc_clear")),
            zero_initialize_workgroup_memory: false,
        });

        let occupancy_gc_repopulate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_occupancy_gc_repopulate_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_occupancy_gc_repopulate")),
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

        let car_transform_shader = world.resource::<GpuCarTransformShader>().0.clone();
        let car_transform_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_car_transform_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: car_transform_shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main")),
            zero_initialize_workgroup_memory: false,
        });

        let car_clear_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("gpu_sim_car_clear_pipeline")),
            layout: vec![layout_desc.clone()], 
            push_constant_ranges: vec![],
            shader: car_transform_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_clear")),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            people_pipeline,
            occupancy_pipeline,
            buildings_pipeline,
            recount_pipeline,
            recalibrate_pipeline,
            occupancy_gc_clear_pipeline,
            occupancy_gc_repopulate_pipeline,
            update_roads_pipeline,
            logic_pipeline,
            spawn_pipeline,
            car_transform_pipeline,
            car_clear_pipeline,
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
    let max_segs = MAX_SEGMENTS as u64; 
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
            size: std::mem::size_of::<GpuStats>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    if b_stats.0.is_none() {
        b_stats.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_building_stats_buffer"),
            size: (MAX_BUILDINGS as u64).max(MIN_GPU_BUFFER_CAPACITY as u64) * 12, // 3 u32 per building (occupants, assigned, tax) = 12 bytes
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    let grid_size = (params.grid_w * params.grid_h) as u64;
    if occupancy.0.is_none() || occupancy.0.as_ref().unwrap().size() < (grid_size * 4) {
        occupancy.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_occupancy_buffer"),
            size: grid_size.max(MIN_GPU_BUFFER_CAPACITY as u64) * 4,
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
    occupancy: Res<GpuOccupancyBuffer>,
    b_stats: Res<GpuBuildingStatsBuffer>,
    path_buffers: Res<crate::compute::gpu_pathfinding::GpuPathBuffers>,
    stats: Res<GpuStatsBuffer>,
    mut bind_group: ResMut<GpuSimBindGroup>,
) {
    let (
        Some(people), 
        Some(roads), 
        Some(buildings), 
        Some(params_buf), 
        Some(path_req_buf), 
        Some(paths_buf), 
        Some(congestion_buf), 
        Some(stats_buf), 
        Some(occupancy_buf), 
        Some(b_stats_buf),
        Some(road_pts),
        Some(elevations),
        Some(transforms)
    ) = (
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
        textures.road_points.as_ref().and_then(|h| gpu_images.get(h)),
        textures.elevations.as_ref().and_then(|h| gpu_images.get(h)),
        textures.car_transforms.as_ref().and_then(|h| gpu_images.get(h)),
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
            road_pts.texture_view.into_binding(),
            elevations.texture_view.into_binding(),
            transforms.texture_view.into_binding(),
            elevations.sampler.into_binding(),
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
        let mut event = crate::TimingEvent::default();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_pipeline = world.resource::<GpuSimPipeline>();
        let bind_group = &world.resource::<GpuSimBindGroup>().0;
        let params = world.resource::<GpuSimParams>();
        let textures = world.resource::<GpuSimTextures>();
        let gpu_images = world.resource::<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>();

        if let (Some(movement_pipe), Some(_occupancy_pipe), Some(logic_pipe), Some(build_pipe), Some(recount_pipe), Some(recalibrate_pipe), Some(gc_clear_pipe), Some(gc_repop_pipe), Some(update_roads_pipe), Some(spawn_pipe), Some(car_transform_pipe), Some(bg)) = (
            pipeline_cache.get_compute_pipeline(gpu_pipeline.people_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.occupancy_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.logic_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.buildings_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.recount_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.recalibrate_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.occupancy_gc_clear_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.occupancy_gc_repopulate_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.update_roads_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.spawn_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.car_transform_pipeline),
            bind_group
        ) {
            // --- OCCUPANCY GARBAGE COLLECTION ---
            // Periodically clear and re-populate the occupancy buffer to prevent "ghost cars"
            if params.do_occupancy_gc > 0 {
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_sim_occupancy_gc_pass"),
                    ..default()
                });
                pass.set_bind_group(0, bg, &[]);
                
                // 1. Clear the entire grid
                pass.set_pipeline(gc_clear_pipe);
                let grid_wg_count = (params.grid_w * params.grid_h + 63) / 64;
                pass.dispatch_workgroups(grid_wg_count, 1, 1);

                // 2. Re-populate with current vehicle positions
                pass.set_pipeline(gc_repop_pipe);
                let p_wg_count = (params.people_count + 63) / 64;
                if p_wg_count > 0 {
                    pass.dispatch_workgroups(p_wg_count, 1, 1);
                }
                drop(pass);
            }

            // Clear recount stats before any logic logic
            // Only clear the first 60 bytes of stats on the first slice of the recount!
            // (The rest of the buffer contains lifetime tax accumulators)
            if params.recount_slice == 0 {
                // First, recalibrate live counter from previous recount result
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_sim_recalibrate_pass"),
                    ..default()
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(recalibrate_pipe);
                pass.dispatch_workgroups(1, 1, 1);
                drop(pass);

                if let Some(b_stats_buf) = world.resource::<GpuBuildingStatsBuffer>().0.as_ref() {
                    render_context.command_encoder().clear_buffer(b_stats_buf, 0, None);
                }
            }

            // --- RECOUNT PASS ---
            // Recount pass (Sliced over 5 frames, 64 people per thread)
            let recount_start_time = std::time::Instant::now();
            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_sim_recount_pass"),
                ..default()
            });
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(recount_pipe);
            
            let slice_count = params.recount_slice_count.max(1);
            let slice_size = (params.people_count + slice_count - 1) / slice_count;
            let chunk_size = 64; 
            let num_threads = (slice_size + chunk_size - 1) / chunk_size;
            let recount_wg_count = (num_threads + 63) / 64;

            if params.people_count > 0 {
                pass.dispatch_workgroups(recount_wg_count, 1, 1);
                event.recount_cycle = Some((params.recount_slice + 1, slice_count));
            }
            drop(pass);
            event.recount = recount_start_time.elapsed().as_secs_f32() * 1000.0;

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

            let p_wg_count = (params.people_count + 63) / 64;

            // End spawning pass
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
            drop(pass);

            // 2b. Car clear/transform pass (every frame)
            if let Some(car_clear_pipe) = pipeline_cache.get_compute_pipeline(gpu_pipeline.car_clear_pipeline) {
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_sim_car_transform_pass"),
                    ..default()
                });
                pass.set_bind_group(0, bg, &[]);

                // First, clear the tail of the transforms texture
                pass.set_pipeline(car_clear_pipe);
                let car_wg_count = (params.car_capacity + 63) / 64;
                if car_wg_count > 0 {
                    pass.dispatch_workgroups(car_wg_count, 1, 1);
                }

                // Then, compute and pack active car transforms
                pass.set_pipeline(car_transform_pipe);
                if p_wg_count > 0 {
                    pass.dispatch_workgroups(p_wg_count, 1, 1);
                }
                drop(pass);
            }

            // 3. People logic pass
            if params.logic_count > 0 {
                let logic_start_time = std::time::Instant::now();
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_sim_logic_pass"),
                    ..default()
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(logic_pipe);
                let logic_wg_count = (params.logic_count + 63) / 64;
                pass.dispatch_workgroups(logic_wg_count, 1, 1);
                drop(pass);
                event.logic = logic_start_time.elapsed().as_secs_f32() * 1000.0;
                event.logic_count = Some(params.logic_count);
                event.logic_cycle = Some((params.logic_start / params.logic_count.max(1) + 1, (params.people_count + params.logic_count.max(1) - 1) / params.logic_count.max(1)));
            }

            // 5. Buildings pass (Updates textures from Recount results)
            if params.b_count > 0 {
                let bldg_start_time = std::time::Instant::now();
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_sim_buildings_pass"),
                    ..default()
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(build_pipe);
                let b_wg_count = (params.b_count + 63) / 64;
                pass.dispatch_workgroups(b_wg_count, 1, 1);
                drop(pass);
                event.bldg = bldg_start_time.elapsed().as_secs_f32() * 1000.0;
                event.bldg_cycle = Some((params.b_start / params.b_count.max(1) + 1, (params.buildings_count + params.b_count.max(1) - 1) / params.b_count.max(1)));
            }

            // 6. Update Roads pass
            if params.r_count > 0 {
                let road_start_time = std::time::Instant::now();
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_sim_roads_pass"),
                    ..default()
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(update_roads_pipe);
                let r_wg_count = (params.r_count + 63) / 64;
                pass.dispatch_workgroups(r_wg_count, 1, 1);
                drop(pass);
                event.road = road_start_time.elapsed().as_secs_f32() * 1000.0;
                event.road_cycle = Some((params.r_start / params.r_count.max(1) + 1, (params.segments_count + params.r_count.max(1) - 1) / params.r_count.max(1)));
            }
        }

        // Copy Stats
        if params.do_stats_readback > 0 {
            let readback_handles = world.resource::<GpuReadbackBufferHandles>();
            let storage_buffers = world.resource::<bevy::render::render_asset::RenderAssets<bevy::render::storage::GpuShaderStorageBuffer>>();
            
            if let (Some(stats_buf), Some(rb_stats_buf)) = (world.resource::<GpuStatsBuffer>().0.as_ref(), storage_buffers.get(readback_handles.stats_buffer.id())) {
                let size = std::mem::size_of::<GpuStats>() as u64;
                render_context.command_encoder().copy_buffer_to_buffer(stats_buf, 0, &rb_stats_buf.buffer, 0, size);
            }
        }

        // Selective Readbacks
        if let Some(selection) = world.get_resource::<crate::ui::inspector::Selection>() {
            let last_copy = world.resource::<GpuLastCopyFrame>();
            let changed = selection.changed_frame != last_copy.0.load(Ordering::Relaxed);
            
            if params.do_inspector_readback > 0 {
                let readback_handles = world.resource::<GpuReadbackBufferHandles>();
                let storage_buffers = world.resource::<bevy::render::render_asset::RenderAssets<bevy::render::storage::GpuShaderStorageBuffer>>();

                match selection.obj {
                    Some(crate::ui::inspector::SelectedObj::Person(pid)) => {
                        if let (Some(people_h), Some(p_buf)) = (textures.people.as_ref(), storage_buffers.get(readback_handles.inspector_p_buf.id())) {
                            if let Some(gpu_img) = gpu_images.get(people_h) {
                                let texel_idx = pid * 3;
                                let x = texel_idx % params.people_tex_w;
                                let y = texel_idx / params.people_tex_w;
                                let mut tex_info = gpu_img.texture.as_image_copy();
                                tex_info.origin = Origin3d { x, y, z: 0 };
                                render_context.command_encoder().copy_texture_to_buffer(
                                    tex_info,
                                    TexelCopyBufferInfo {
                                        buffer: &p_buf.buffer,
                                        layout: TexelCopyBufferLayout {
                                            offset: 0,
                                            bytes_per_row: Some(256),
                                            rows_per_image: None,
                                        },
                                    },
                                    Extent3d { width: 3, height: 1, depth_or_array_layers: 1 },
                                );
                                last_copy.0.store(selection.changed_frame, Ordering::Relaxed);
                            }
                        }
                    }
                    Some(crate::ui::inspector::SelectedObj::Building(bid)) => {
                        if let (Some(buildings_h), Some(b_buf)) = (textures.buildings.as_ref(), storage_buffers.get(readback_handles.inspector_b_buf.id())) {
                            if let Some(gpu_img) = gpu_images.get(buildings_h) {
                                let texel_idx = bid * 3;
                                let x = texel_idx % params.buildings_tex_w;
                                let y = texel_idx / params.buildings_tex_w;
                                let mut tex_info = gpu_img.texture.as_image_copy();
                                tex_info.origin = Origin3d { x, y, z: 0 };
                                render_context.command_encoder().copy_texture_to_buffer(
                                    tex_info,
                                    TexelCopyBufferInfo {
                                        buffer: &b_buf.buffer,
                                        layout: TexelCopyBufferLayout {
                                            offset: 0,
                                            bytes_per_row: Some(256),
                                            rows_per_image: None,
                                        },
                                    },
                                    Extent3d { width: 3, height: 1, depth_or_array_layers: 1 },
                                );
                                last_copy.0.store(selection.changed_frame, Ordering::Relaxed);
                            }
                        }
                    }
                    _ => {
                        if changed {
                            last_copy.0.store(selection.changed_frame, Ordering::Relaxed);
                        }
                    }
                }
            } else if changed {
                 last_copy.0.store(selection.changed_frame, Ordering::Relaxed);
            }
        }

        if let Ok(tx) = world.resource::<crate::TimingsSender>().0.lock() {
            event.total_compute = start.elapsed().as_secs_f32() * 1000.0;
            
            // Collect occupancy data
            event.occ_people = Some((params.people_count, params.max_people));
            event.occ_bldgs = Some((params.buildings_count, params.max_buildings));
            event.occ_segments = Some((params.segments_count, params.max_segments));

            let _ = tx.send(event);
        }

        Ok(())
    }
}

fn request_gpu_readback(
    mut commands: Commands,
    params: Res<GpuSimParams>,
    selection: Option<Res<crate::ui::inspector::Selection>>,
    mut last_copy_frame: Local<u32>,
    readback_buffers: Res<GpuReadbackBufferHandles>,
    mut last_readback_frame: Local<u32>,
) {
    if params.do_stats_readback > 0 {
        if *last_readback_frame != params.recount_slice {
            commands.spawn(bevy::render::gpu_readback::Readback::buffer(readback_buffers.stats_buffer.clone()))
                .observe(|trigger: bevy::ecs::observer::On<bevy::render::gpu_readback::ReadbackComplete>, mut counters: ResMut<crate::sim::counters::SimCounters>| {
                    let data = &trigger.event().data;
                    if data.len() >= std::mem::size_of::<GpuStats>() {
                        let stats: GpuStats = *bytemuck::from_bytes(&data[..std::mem::size_of::<GpuStats>()]);
                        counters.people = stats.people_count;
                        counters.cars = stats.live_car_count;
                        counters.bankrupt = stats.bankrupt_count;
                        counters.res_occupants = stats.home_count;
                        counters.office_occupants = stats.work_count;
                        counters.shop_occupants = stats.shop_count;
                        counters.tax_income_total = stats.tax_income_total;
                        counters.tax_rent_total = stats.tax_rent_total;
                        counters.tax_consumption_total = stats.tax_consumption_total;
                        counters.money_total = stats.total_money;
                    }
                });
            *last_readback_frame = params.recount_slice;
        }
    } else {
        *last_readback_frame = params.recount_slice;
    }

    if let Some(sel) = selection {
        let changed = sel.changed_frame != *last_copy_frame;
        if changed && params.do_inspector_readback > 0 {
            match sel.obj {
                Some(crate::ui::inspector::SelectedObj::Person(pid)) => {
                    commands.spawn(bevy::render::gpu_readback::Readback::buffer(readback_buffers.inspector_p_buf.clone()))
                        .observe(move |trigger: bevy::ecs::observer::On<bevy::render::gpu_readback::ReadbackComplete>, mut people: ResMut<crate::sim::people::PeopleData>| {
                            let data = &trigger.event().data;
                            if data.len() >= std::mem::size_of::<crate::sim::people::PersonRow>() {
                                let row: crate::sim::people::PersonRow = *bytemuck::from_bytes(&data[..std::mem::size_of::<crate::sim::people::PersonRow>()]);
                                if (pid as usize) < people.rows.len() {
                                    people.rows[pid as usize] = row;
                                }
                            }
                        });
                }
                Some(crate::ui::inspector::SelectedObj::Building(bid)) => {
                    commands.spawn(bevy::render::gpu_readback::Readback::buffer(readback_buffers.inspector_b_buf.clone()))
                        .observe(move |trigger: bevy::ecs::observer::On<bevy::render::gpu_readback::ReadbackComplete>, mut buildings: ResMut<BuildingData>, mut grid: ResMut<crate::sim::grid::CityGrid>, mut counters: ResMut<crate::sim::counters::SimCounters>| {
                            let data = &trigger.event().data;
                            if data.len() >= std::mem::size_of::<crate::sim::buildings::BuildingRow>() {
                                let r: crate::sim::buildings::BuildingRow = *bytemuck::from_bytes(&data[..std::mem::size_of::<crate::sim::buildings::BuildingRow>()]);
                                if let Some(b) = buildings.items.get_mut(bid as usize) {
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
                                            if current_bid == bid {
                                                grid.set(b.tile.0, b.tile.1, crate::sim::grid::Tile::Zone(b.btype));
                                                counters.destroyed_buildings += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        });
                }
                _ => {}
            }
            *last_copy_frame = sel.changed_frame;
        }
    }
}

pub fn update_gpu_sim_params(
    time: &Time,
    config: &SimConfig,
    settings: &SimSettings,
    people: &PeopleData,
    buildings: &BuildingData,
    roads: &crate::sim::roads::RoadData,
    grid: &crate::sim::grid::CityGrid,
    gpu_params: &mut GpuSimParams,
    entry_seg: u32,
    car_capacity: u32,
) {
    gpu_params.dt = time.delta_secs();
    gpu_params.home_duration = config.home;
    gpu_params.work_duration = config.work;
    gpu_params.shop_duration = config.shop;
    gpu_params.home_to_work_prob = config.home_to_work_prob;
    gpu_params.people_count = people.len;
    gpu_params.people_tex_w = people.tex_width;
    gpu_params.people_tex_h = people.tex_height;
    gpu_params.rng_seed = rand::random();
    gpu_params.abandon_multiplier = settings.abandon_multiplier;
    gpu_params.rent_cost = settings.rent_cost;
    gpu_params.work_salary = settings.work_salary;
    gpu_params.shop_cost = settings.shop_cost;
    gpu_params.tax_income = settings.tax_income;
    gpu_params.tax_rent = settings.tax_rent;
    gpu_params.tax_consumption = settings.tax_consumption;
    gpu_params.buildings_tex_w = buildings.tex_width;
    gpu_params.buildings_tex_h = buildings.tex_height;
    gpu_params.roads_tex_w = roads.tex_width;
    gpu_params.buildings_count = buildings.items.len() as u32;
    gpu_params.segments_count = roads.segments.len() as u32;
    gpu_params.grid_w = grid.width;
    gpu_params.grid_h = grid.height;
    gpu_params.entry_seg = entry_seg;
    gpu_params.collisions_enabled = settings.collisions_enabled;
    gpu_params.car_capacity = car_capacity;
    gpu_params.max_people = MAX_PEOPLE;
    gpu_params.max_segments = MAX_SEGMENTS;
    gpu_params.max_buildings = MAX_BUILDINGS;
    gpu_params.max_path_len = MAX_PATH_LEN;
    gpu_params.max_path_requests = MAX_PATH_REQUESTS;
}
