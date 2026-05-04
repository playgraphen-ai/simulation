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
    segments_count: u32, // Passed via _pad0 in Rust
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(1) var roads_tex : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(6) var<storage, read_write> congestion_buffer: array<atomic<u32>>;

fn road_coords(sid: u32) -> array<vec2<i32>, 2> {
    let base = i32(sid * 2u);
    let w = i32(params.roads_tex_w);
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    return array<vec2<i32>, 2>(c0, c1);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg_id = gid.x;
    if seg_id >= params.segments_count { return; }

    // Read and reset the congestion bits atomically in one instruction
    let bits_set = atomicExchange(&congestion_buffer[seg_id], 0u);
    
    // Count population of set bits
    let n = countOneBits(bits_set);
    
    // Probabilistic estimation formula to compensate for collisions.
    // If n=32 (saturated), clamp to 31 to prevent division by zero or infinite log.
    let safe_n = min(f32(n), 31.0);
    let estimated_cars = -32.0 * log(1.0 - (safe_n / 32.0));

    // Calculate congestion factor.
    // Let's assume a segment heavily saturated at ~20 cars.
    let max_cars_on_segment = 20.0;
    
    // Factor: 1.0 = empty (full speed), 0.1 = completely jammed.
    let congestion_factor = max(0.1, 1.0 - (estimated_cars / max_cars_on_segment));

    // Load current speed data
    let coords = road_coords(seg_id);
    var tex1 = textureLoad(roads_tex, coords[1]);
    
    // EMA (Exponential Moving Average) for smoothing: 95% old, 5% new
    tex1.x = tex1.x * 0.95 + congestion_factor * 0.05;
    
    // Store updated speed back to the texture
    textureStore(roads_tex, coords[1], tex1);
}
