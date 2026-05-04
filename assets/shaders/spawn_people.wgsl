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
    buildings_tex_w: u32,
    buildings_tex_h: u32,
    roads_tex_w: u32,
    buildings_count: u32,
    segments_count: u32,
    spawn_count: u32,
    spawn_start_index: u32,
};

@group(0) @binding(0) var people_tex    : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var roads_tex     : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var buildings_tex : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> params: SimParams;
// Binding 4: Path queue (unused here)
// Binding 5: Path buffer (unused here)
// Binding 6: Congestion buffer (unused here)

fn person_coords(pid: u32) -> array<vec2<i32>, 3> {
    let base = i32(pid * 3u);
    let w = i32(params.people_tex_w);
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn building_coords(bid: u32) -> array<vec2<i32>, 3> {
    let base = i32(bid * 3u);
    let w = i32(params.buildings_tex_w);
    let c0 = vec2<i32>(base % w, base / w);
    let c1 = vec2<i32>((base + 1) % w, (base + 1) / w);
    let c2 = vec2<i32>((base + 2) % w, (base + 2) / w);
    return array<vec2<i32>, 3>(c0, c1, c2);
}

fn rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let request_idx = gid.x;
    if request_idx >= params.spawn_count { return; }

    let pid = params.spawn_start_index + request_idx;
    var rng_state = params.rng_seed + pid + 777u;

    // 1. Find Home (Residential)
    var home_id = 0u;
    var found_home = false;
    for (var i = 0u; i < 50u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let coords = building_coords(bid);
        let tex0 = textureLoad(buildings_tex, coords[0]);
        var tex1 = textureLoad(buildings_tex, coords[1]);
        
        let btype = tex0.z; // 0=Res, 1=Off, 2=Shop
        let occupants = tex1.y;
        let capacity = tex1.z;
        
        if btype == 0.0 && occupants < capacity {
            home_id = bid;
            found_home = true;
            
            // Increment occupants (probabilistic, non-atomic for now to avoid complexity, 
            // but retries help. For thousands of spawns, some overlap is okay 
            // as buildings will auto-correct next frame)
            tex1.y = tex1.y + 1.0;
            textureStore(buildings_tex, coords[1], tex1);
            break;
        }
    }
    
    if !found_home { return; } // Failed to find home, slot stays dead

    // 2. Find Work (Office)
    var work_id = home_id;
    var found_work = false;
    for (var i = 0u; i < 50u; i = i + 1u) {
        let bid = u32(rand(&rng_state) * f32(params.buildings_count));
        let coords = building_coords(bid);
        let tex0 = textureLoad(buildings_tex, coords[0]);
        let tex1 = textureLoad(buildings_tex, coords[1]);
        
        let btype = tex0.z;
        let occupants = tex1.y;
        let capacity = tex1.z;
        
        if btype == 1.0 && occupants < capacity {
            work_id = bid;
            found_work = true;
            break;
        }
    }

    // 3. Initialize Person
    let p_coords = person_coords(pid);
    let home_coords = building_coords(home_id);
    let h_tex1 = textureLoad(buildings_tex, home_coords[1]);
    let home_seg = h_tex1.w;

    let money = 50.0 + rand(&rng_state) * 450.0;
    let age = 18.0 + rand(&rng_state) * 57.0;

    // activity_code: 0.0 = Travel (ACT_TRAVEL)
    // activity_time: -10.0 = Waiting for path timeout
    let texel0 = vec4<f32>(money, age, f32(work_id), f32(home_id));
    let texel1 = vec4<f32>(f32(work_id), 0.0, -10.0, 0.0);
    let texel2 = vec4<f32>(home_seg, home_seg, 0.0, 0.0);

    textureStore(people_tex, p_coords[0], texel0);
    textureStore(people_tex, p_coords[1], texel1);
    textureStore(people_tex, p_coords[2], texel2);
}
