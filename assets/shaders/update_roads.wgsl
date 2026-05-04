// Compute shader: Probabilistic congestion estimation
//
// Calculates estimated car density using random atomicOr bits
// and updates the road speeds (exponential moving average).
//
// Bindings:
//   @group(0) @binding(0) var roads_tex : texture_storage_2d<rgba32float, read_write>;
//   @group(0) @binding(1) var<storage, read_write> congestion_buffer: array<atomic<u32>>;
//   @group(0) @binding(2) var<uniform> params: GpuSimParams;

struct SimParams {
    dt: f32,
    home_duration: f32,
    work_duration: f32,
    shop_duration: f32,
    home_to_work_prob: f32,
    people_count: u32,
    people_tex_w: u32,
    people_tex_h: u32,
    rng_seed: u32,
    abandon_multiplier: f32,
    rent_cost: f32,
    work_salary: f32,
    shop_cost: f32,
    buildings_tex_w: u32,
    buildings_tex_h: u32,
    roads_tex_w: u32,
    buildings_count: u32,
    segments_count: u32,
    spawn_count: u32,
    spawn_start_index: u32,
    b_start: u32,
    b_count: u32,
    logic_start: u32,
    logic_count: u32,
    r_start: u32,
    r_count: u32,
    cycle_frames: u32,
    do_readback: u32,
    reset_stats: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct PathRequest {
    start: u32,
    target_seg: u32,
    person_id: u32,
};

struct PathRequestQueue {
    count_x: atomic<u32>,
    count_y: u32,
    count_z: u32,
    processed: atomic<u32>,
    requests: array<PathRequest>,
};

struct GpuStats {
    people_count: atomic<u32>,
    home_count: atomic<u32>,
    work_count: atomic<u32>,
    shop_count: atomic<u32>,
    travelling_count: atomic<u32>,
    total_money: atomic<u32>,
    residential_occupancy: atomic<u32>,
    office_occupancy: atomic<u32>,
    shop_occupancy: atomic<u32>,
    residential_count: atomic<u32>,
    office_count: atomic<u32>,
    shop_count_b: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var people_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var buildings_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<storage, read_write> path_queue: PathRequestQueue;
@group(0) @binding(5) var<storage, read_write> person_paths: array<u32>;
@group(0) @binding(6) var<storage, read_write> congestion_buffer: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> stats: GpuStats;

fn road_coords(sid: u32) -> array<vec2<i32>, 2> {
    let base = i32(sid * 2u);
    let w = max(1, i32(params.roads_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    return array<vec2<i32>, 2>(c0, c1);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg_id = gid.x + params.r_start;
    if seg_id >= params.r_start + params.r_count || seg_id >= params.segments_count { return; }

    // Read and reset the accumulated counts atomically
    let total_cars = atomicExchange(&congestion_buffer[seg_id], 0u);
    
    let estimated_cars = f32(total_cars) / f32(params.cycle_frames);

    let max_cars_on_segment = 20.0;
    
    // Factor: 1.0 = empty (full speed), 0.1 = completely jammed.
    let congestion_factor = max(0.1, 1.0 - (estimated_cars / max_cars_on_segment));

    // Load current speed data
    let coords = road_coords(seg_id);
    var tex1 = textureLoad(roads_tex, coords[1]);
    
    // EMA (Exponential Moving Average) for smoothing: 50% old, 50% new
    // Since it only runs once every cycle (e.g. 90 frames), we weight the new reading much more.
    tex1.x = tex1.x * 0.5 + congestion_factor * 0.5;
    
    // Store updated speed back to the texture
    textureStore(roads_tex, coords[1], tex1);
}
