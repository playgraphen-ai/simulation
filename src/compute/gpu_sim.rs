use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderContext},
        Render, RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{RenderLabel, NodeRunError, RenderGraphContext},
    },
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender};

use crate::sim::people::PeopleData;
use crate::sim::buildings::BuildingData;
use crate::sim::{ActivityDurations, SimSettings};

#[derive(Resource, Default)]
pub struct GpuReadbackBuffer {
    pub people_buffer: Option<Buffer>,
    pub buildings_buffer: Option<Buffer>,
    pub people_size: u64,
    pub buildings_size: u64,
}

fn prepare_readback_buffers(
    render_device: Res<RenderDevice>,
    params: Res<GpuSimParams>,
    mut readback: ResMut<GpuReadbackBuffer>,
) {
    let unaligned_p_row = params.people_tex_w * 16;
    let align = 256;
    let aligned_p_row = (unaligned_p_row + align - 1) & !(align - 1);
    let p_size = (aligned_p_row * params.people_tex_h) as u64;

    if p_size > 0 && readback.people_size != p_size {
        readback.people_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_people_readback_buffer"),
            size: p_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        readback.people_size = p_size;
    }

    let mut b_w = 128; // fallback
    if params.buildings_tex_w > 0 { b_w = params.buildings_tex_w; }
    let unaligned_b_row = b_w * 16;
    let aligned_b_row = (unaligned_b_row + align - 1) & !(align - 1);
    let mut b_h = 128; // fallback
    if params.buildings_tex_h > 0 { b_h = params.buildings_tex_h; }
    let b_size = (aligned_b_row * b_h) as u64;

    if b_size > 0 && readback.buildings_size != b_size {
        readback.buildings_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_buildings_readback_buffer"),
            size: b_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        readback.buildings_size = b_size;
    }
}

pub struct GpuSimPlugin;

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct GpuSimLabel;

#[derive(Resource)]
pub struct PeopleReceiver(pub Mutex<Receiver<Vec<crate::sim::people::PersonRow>>>);

#[derive(Resource)]
pub struct BuildingsReceiver(pub Mutex<Receiver<Vec<crate::sim::buildings::BuildingRow>>>);

#[derive(Resource)]
pub struct PeopleSender(pub Mutex<Sender<Vec<crate::sim::people::PersonRow>>>);

#[derive(Resource)]
pub struct BuildingsSender(pub Mutex<Sender<Vec<crate::sim::buildings::BuildingRow>>>);

#[derive(Resource)]
struct GpuSimShader(Handle<Shader>);

#[derive(Resource)]
struct GpuUpdateRoadsShader(Handle<Shader>);

impl Plugin for GpuSimPlugin {
    fn build(&self, app: &mut App) {
        let shader = app.world_mut().resource::<AssetServer>().load("shaders/sim_people.wgsl");
        let update_roads_shader = app.world_mut().resource::<AssetServer>().load("shaders/update_roads.wgsl");
        let (tx_p, rx_p) = std::sync::mpsc::channel();
        let (tx_b, rx_b) = std::sync::mpsc::channel();
        
        app.insert_resource(PeopleReceiver(Mutex::new(rx_p)));
        app.insert_resource(BuildingsReceiver(Mutex::new(rx_b)));

        app.add_plugins(ExtractResourcePlugin::<GpuSimParams>::default())
           .add_plugins(ExtractResourcePlugin::<GpuSimTextures>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(GpuSimShader(shader))
            .insert_resource(GpuUpdateRoadsShader(update_roads_shader))
            .insert_resource(PeopleSender(Mutex::new(tx_p)))
            .insert_resource(BuildingsSender(Mutex::new(tx_b)))
            .init_resource::<GpuReadbackBuffer>()
            .init_resource::<GpuSimBindGroup>()
            .init_resource::<GpuSimUniformBuffer>()
            .init_resource::<GpuCongestionBuffer>()
            .add_systems(Render, (
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
    pub _pad: u32,
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
            _pad: 0,
        }
    }
}

#[derive(Resource)]
struct GpuSimPipeline {
    pub people_pipeline: CachedComputePipelineId,
    pub logic_pipeline: CachedComputePipelineId,
    pub buildings_pipeline: CachedComputePipelineId,
    pub update_roads_pipeline: CachedComputePipelineId,
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
            shader: shader,
            shader_defs: vec![],
            entry_point: Some(Cow::Borrowed("main_buildings")),
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

        Self {
            people_pipeline,
            buildings_pipeline,
            update_roads_pipeline,
            logic_pipeline,
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

fn prepare_gpu_sim_buffers(
    render_device: Res<RenderDevice>,
    params: Res<GpuSimParams>,
    mut buffer: ResMut<GpuSimUniformBuffer>,
    mut congestion: ResMut<GpuCongestionBuffer>,
) {
    let bytes = bytemuck::bytes_of(&*params);
    let b = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gpu_sim_params_buffer"),
        contents: bytes,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    buffer.0 = Some(b);

    // Congestion buffer: segments_count * 4 bytes
    // We allocate a sufficiently large buffer or resize it if needed.
    let required_size = (params.segments_count as u64 * 4).max(4);
    let max_segs = 65536u64; // Constante MAX_SEGMENTS
    if congestion.0.is_none() || congestion.0.as_ref().unwrap().size() < (max_segs * 4) {
        congestion.0 = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("gpu_congestion_buffer"),
            size: max_segs * 4,
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
    path_buffers: Res<crate::compute::gpu_pathfinding::GpuPathBuffers>,
    mut bind_group: ResMut<GpuSimBindGroup>,
) {
    let (Some(people), Some(roads), Some(buildings), Some(params_buf), Some(path_req_buf), Some(paths_buf), Some(congestion_buf)) = (
        textures.people.as_ref().and_then(|h| gpu_images.get(h)),
        textures.roads.as_ref().and_then(|h| gpu_images.get(h)),
        textures.buildings.as_ref().and_then(|h| gpu_images.get(h)),
        buffer.0.as_ref(),
        path_buffers.requests.as_ref(),
        path_buffers.paths.as_ref(),
        congestion.0.as_ref(),
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
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_pipeline = world.resource::<GpuSimPipeline>();
        let bind_group = &world.resource::<GpuSimBindGroup>().0;
        let params = world.resource::<GpuSimParams>();
        let textures = world.resource::<GpuSimTextures>();
        let gpu_images = world.resource::<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>();
        let readback = world.resource::<GpuReadbackBuffer>();

        if let (Some(movement_pipe), Some(logic_pipe), Some(build_pipe), Some(update_roads_pipe), Some(bg)) = (
            pipeline_cache.get_compute_pipeline(gpu_pipeline.people_pipeline), // This is movement now
            pipeline_cache.get_compute_pipeline(gpu_pipeline.logic_pipeline), // This is logic
            pipeline_cache.get_compute_pipeline(gpu_pipeline.buildings_pipeline),
            pipeline_cache.get_compute_pipeline(gpu_pipeline.update_roads_pipeline),
            bind_group
        ) {
            let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("gpu_sim_pass"),
                ..default()
            });
            pass.set_bind_group(0, bg, &[]);
            
            // 1. Buildings pass
            if params.b_count > 0 {
                pass.set_pipeline(build_pipe);
                let b_wg_count = (params.b_count + 63) / 64;
                pass.dispatch_workgroups(b_wg_count, 1, 1);
            }

            // 2. People movement pass (every frame)
            pass.set_pipeline(movement_pipe);
            let p_wg_count = (params.people_count + 63) / 64;
            if p_wg_count > 0 {
                pass.dispatch_workgroups(p_wg_count, 1, 1);
            }
            
            // 3. People logic pass
            if params.logic_count > 0 {
                pass.set_pipeline(logic_pipe);
                let logic_wg_count = (params.logic_count + 63) / 64;
                pass.dispatch_workgroups(logic_wg_count, 1, 1);
            }

            // 4. Update Roads pass
            if params.r_count > 0 {
                pass.set_pipeline(update_roads_pipe);
                let r_wg_count = (params.r_count + 63) / 64;
                pass.dispatch_workgroups(r_wg_count, 1, 1);
            }
        }

        // Copy People
        if let (Some(people_h), Some(p_buf)) = (textures.people.as_ref(), readback.people_buffer.as_ref()) {
            if let Some(gpu_img) = gpu_images.get(people_h) {
                render_context.command_encoder().copy_texture_to_buffer(
                    gpu_img.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer: p_buf,
                        layout: TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some({
                                let unaligned = gpu_img.texture.width() * 16;
                                let align = 256;
                                (unaligned + align - 1) & !(align - 1)
                            }),
                            rows_per_image: None,
                        },
                    },
                    gpu_img.texture.size(),
                );
            }
        }

        // Copy Buildings
        if let (Some(buildings_h), Some(b_buf)) = (textures.buildings.as_ref(), readback.buildings_buffer.as_ref()) {
            if let Some(gpu_img) = gpu_images.get(buildings_h) {
                render_context.command_encoder().copy_texture_to_buffer(
                    gpu_img.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer: b_buf,
                        layout: TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some({
                                let unaligned = gpu_img.texture.width() * 16;
                                let align = 256;
                                (unaligned + align - 1) & !(align - 1)
                            }),
                            rows_per_image: None,
                        },
                    },
                    gpu_img.texture.size(),
                );
            }
        }

        Ok(())
    }
}

fn map_and_send_readback(
    render_device: Res<RenderDevice>,
    readback: Res<GpuReadbackBuffer>,
    sender_p: Res<PeopleSender>,
    sender_b: Res<BuildingsSender>,
    params: Res<GpuSimParams>,
) {
    let (Some(p_buf), Some(b_buf)) = (readback.people_buffer.as_ref(), readback.buildings_buffer.as_ref()) else { return; };
    
    let tx_p = sender_p.0.lock().unwrap().clone();
    let p_clone = p_buf.clone();
    let p_w = params.people_tex_w;
    let p_h = params.people_tex_h;
    p_buf.slice(..).map_async(MapMode::Read, move |res_p| {
        if res_p.is_ok() {
            let data_p = p_clone.slice(..).get_mapped_range();
            let unaligned_row = p_w as usize * 16;
            let align = 256;
            let aligned_row = (unaligned_row + align - 1) & !(align - 1);
            
            let mut rows_p = Vec::with_capacity((p_w * p_h / 3) as usize);
            for y in 0..p_h as usize {
                let start = y * aligned_row;
                let end = start + unaligned_row;
                if end <= data_p.len() {
                    let row_data = &data_p[start..end];
                    let persons: &[crate::sim::people::PersonRow] = bytemuck::cast_slice(row_data);
                    rows_p.extend_from_slice(persons);
                }
            }

            drop(data_p);
            p_clone.unmap();
            let _ = tx_p.send(rows_p);
        }
    });

    let tx_b = sender_b.0.lock().unwrap().clone();
    let b_clone = b_buf.clone();
    let b_w = params.buildings_tex_w;
    let b_h = params.buildings_tex_h;
    b_buf.slice(..).map_async(MapMode::Read, move |res_b| {
        if res_b.is_ok() {
            let data_b = b_clone.slice(..).get_mapped_range();
            let unaligned_row = b_w as usize * 16;
            let align = 256;
            let aligned_row = (unaligned_row + align - 1) & !(align - 1);
            
            let mut rows_b = Vec::with_capacity((b_w * b_h / 3) as usize);
            for y in 0..b_h as usize {
                let start = y * aligned_row;
                let end = start + unaligned_row;
                if end <= data_b.len() {
                    let row_data = &data_b[start..end];
                    let buildings: &[crate::sim::buildings::BuildingRow] = bytemuck::cast_slice(row_data);
                    rows_b.extend_from_slice(buildings);
                }
            }

            drop(data_b);
            b_clone.unmap();
            let _ = tx_b.send(rows_b);
        }
    });
    
    let _ = render_device.poll(bevy::render::render_resource::PollType::wait_indefinitely());
}

pub fn apply_gpu_readback(
    rx_p: Res<PeopleReceiver>,
    rx_b: Res<BuildingsReceiver>,
    mut people: ResMut<PeopleData>,
    mut buildings: ResMut<BuildingData>,
    mut grid: ResMut<crate::sim::grid::CityGrid>,
    mut counters: ResMut<crate::sim::counters::SimCounters>,
) {
    if let Ok(rx) = rx_p.0.lock() {
        while let Ok(p_rows) = rx.try_recv() {
            let np = (people.len as usize).min(p_rows.len());
            for i in 0..np {
                people.rows[i] = p_rows[i];
            }
        }
    }
    if let Ok(rx) = rx_b.0.lock() {
        while let Ok(b_rows) = rx.try_recv() {
            let nb = (buildings.items.len()).min(b_rows.len());
            for i in 0..nb {
                let b = &mut buildings.items[i];
                let r = &b_rows[i];
                
                b.occupants = r.occupants as u32;
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
                        if current_bid == i as u32 {
                            grid.set(b.tile.0, b.tile.1, crate::sim::grid::Tile::Zone(b.btype));
                            counters.destroyed_buildings += 1;
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
    pending: ResMut<crate::compute::spawn::PendingGpuSpawns>,
    gpu_params: &mut GpuSimParams,
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

    if pending.count > 0 {
        gpu_params.spawn_count = pending.count;
        gpu_params.spawn_start_index = people.len.saturating_sub(pending.count);
        // On garde pending.count pour l'extraction, et on l'efface
        // via un système différé pour s'assurer que le RenderApp l'a copié.
    } else {
        gpu_params.spawn_count = 0;
        gpu_params.spawn_start_index = 0;
    }
}

pub fn clear_pending_spawns(mut pending: ResMut<crate::compute::spawn::PendingGpuSpawns>, mut frame_count: Local<u32>) {
    // Wait for at least 1 frame so RenderApp is guaranteed to extract it.
    if pending.count > 0 {
        *frame_count += 1;
        if *frame_count > 1 {
            pending.count = 0;
            *frame_count = 0;
        }
    } else {
        *frame_count = 0;
    }
}
