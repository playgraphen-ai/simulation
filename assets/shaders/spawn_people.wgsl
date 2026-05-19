// Compute shader: parallel spawn of people.
//
// Finds a home (Residential) and work (Office) for each new person.
// Uses probabilistic search with retries to find buildings with capacity.

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
    shop_food_gain: f32,
    tax_income: f32,
    tax_rent: f32,
    tax_consumption: f32,
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
    do_stats_readback: u32,
    reset_stats: u32,
    grid_w: u32,
    grid_h: u32,
    entry_seg: u32,
    collisions_enabled: f32,
    recount_slice: u32,
    recount_slice_count: u32,
    do_occupancy_gc: u32,
    do_inspector_readback: u32,
    car_capacity: u32,
    max_people: u32,
    max_segments: u32,
    max_buildings: u32,
    max_path_len: u32,
    max_path_requests: u32,
    sim_time: f32,
};

@group(0) @binding(0) var people_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var buildings_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<storage, read_write> path_queue: array<u32>;
@group(0) @binding(5) var<storage, read_write> person_paths: array<u32>;
@group(0) @binding(9) var<storage, read_write> building_stats: array<atomic<u32>>;

const ACT_TRAVEL: f32 = 0.0;
const ACT_HOME:   f32 = 1.0;
const ACT_WORK:   f32 = 2.0;
const ACT_SHOP:   f32 = 3.0;

fn rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x) / 4294967296.0;
}

fn person_coords(pid: u32) -> array<vec2<i32>, 4> {
    let base = i32(pid * 4u);
    let w = max(1, i32(params.people_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    let c3 = vec2<i32>((base + 3) % w, (base + 3) / w);
    return array<vec2<i32>, 4>(c0, c1, c2, c3);
}

fn building_coords(bid: u32) -> array<vec2<i32>, 3> {
    let base = i32(bid * 3u);
    let w = max(1, i32(params.buildings_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn road_coords(sid: u32) -> array<vec2<i32>, 5> {
    let base = i32(sid * 5u);
    let w = max(1, i32(params.roads_tex_w));
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    let c3 = vec2<i32>((base + 3) % w, (base + 3) / w);
    let c4 = vec2<i32>((base + 4) % w, (base + 4) / w);
    return array<vec2<i32>, 5>(c0, c1, c2, c3, c4);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_idx = gid.x;
    if spawn_idx >= params.spawn_count { return; }

    let pid = params.spawn_start_index + spawn_idx;
    if pid >= params.max_people { return; }

    var rng_state = params.rng_seed + pid + 123u;

    if params.buildings_count == 0u { return; }

    // 1. Find a Home (Residential building with capacity)
    var home_id = 0u;
    var found_home = false;
    for (var i = 0u; i < 20u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let b_coords = building_coords(bid);
        let b_tex0 = textureLoad(buildings_tex, b_coords[0]);
        let b_tex1 = textureLoad(buildings_tex, b_coords[1]);
        
        let assigned_at = atomicLoad(&building_stats[bid * 3u + 1u]);
        if b_tex0.z == 0.0 && f32(assigned_at) < b_tex1.z { // Residential and has space
            let home_seg = u32(b_tex1.w);
            let r_coords = road_coords(home_seg);
            let r_tex1 = textureLoad(roads_tex, r_coords[1]);
            if r_tex1.x <= 0.15 { continue; } // Road is full, try another or wait

            home_id = bid;
            found_home = true;
            atomicAdd(&building_stats[bid * 3u + 1u], 1u); // Increment assigned
            break;
        }
    }

    if !found_home { return; }

    // 2. Find a Work (Office building with capacity)
    var work_id = home_id; // Default to stay at home if no job found
    for (var i = 0u; i < 20u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let b_coords = building_coords(bid);
        let b_tex0 = textureLoad(buildings_tex, b_coords[0]);
        let b_tex1 = textureLoad(buildings_tex, b_coords[1]);

        let assigned_at = atomicLoad(&building_stats[bid * 3u + 1u]);
        if b_tex0.z == 1.0 && f32(assigned_at) < b_tex1.z { // Office and has space
            work_id = bid;
            atomicAdd(&building_stats[bid * 3u + 1u], 1u); // Increment assigned
            break;
        }
    }

    // 3. Initialize Person Data
    let home_coords = building_coords(home_id);
    let h_tex1 = textureLoad(buildings_tex, home_coords[1]);
    let h_tex2 = textureLoad(buildings_tex, home_coords[2]);
    let home_seg = h_tex1.w;
    let home_t = h_tex2.z;

    let time_since_rent = rand(&rng_state) * 300.0;
    
    // Texel 0: [money, reserved_idx, destination, home]
    let texel0 = vec4<f32>(50.0 + rand(&rng_state) * 450.0, -1.0, f32(home_id), f32(home_id));
    // Texel 1: [work, activity_code, activity_time, path_cursor]
    let texel1 = vec4<f32>(f32(work_id), ACT_HOME, params.home_duration * rand(&rng_state), 0.0);
    // Texel 2: [current_seg, prev_seg, start_t, target_t]
    let texel2 = vec4<f32>(home_seg, home_seg, home_t, 0.0);
    // Texel 3: [food_stock, _pad, _pad, _pad]
    let texel3 = vec4<f32>(10.0 + rand(&rng_state) * 90.0, 0.0, 0.0, 0.0);

    let p_coords = person_coords(pid);
    textureStore(people_tex, p_coords[0], texel0);
    textureStore(people_tex, p_coords[1], texel1);
    textureStore(people_tex, p_coords[2], texel2);
    textureStore(people_tex, p_coords[3], texel3);

    // Reset path buffer for this person
    let path_idx = pid * params.max_path_len;
    if path_idx < params.max_people * params.max_path_len {
        person_paths[path_idx] = 0xFFFFFFFFu;
    }
}
