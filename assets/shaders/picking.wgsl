struct PickingParams {
    ray_origin: vec4<f32>,
    ray_dir: vec4<f32>,
    is_active: u32,
    max_cars: u32,
    transforms_w: u32,
    pad: u32,
}

struct PickingResult {
    id: atomic<u32>,
    dist: atomic<u32>, // bitcast f32
}

@group(0) @binding(0) var transforms_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: PickingParams;
@group(0) @binding(2) var<storage, read_write> result: PickingResult;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pid = global_id.x;

    // Initialization block on thread 0
    if pid == 0u {
        atomicStore(&result.id, 0xFFFFFFFFu);
        atomicStore(&result.dist, bitcast<u32>(1000000.0));
    }

    if pid >= params.max_cars { return; }

    let t_w = params.transforms_w;
    let t_idx = pid * 4u;
    let tx0 = t_idx % t_w;
    let ty0 = t_idx / t_w;

    let pos_col = textureLoad(transforms_tex, vec2<i32>(i32(tx0), i32(ty0)), 0);
    let world_pos = pos_col.xyz;

    if world_pos.y < -1000.0 { return; } // Inactive/hidden car

    // Calculate distance from ray to point
    let w = world_pos - params.ray_origin.xyz;
    let dist_along_ray = dot(w, params.ray_dir.xyz);
    
    // Only consider cars in front of the camera
    if dist_along_ray < 0.0 { return; }

    let projected_pos = params.ray_origin.xyz + params.ray_dir.xyz * dist_along_ray;
    let dist_to_ray_sq = dot(world_pos - projected_pos, world_pos - projected_pos);

    let click_radius_sq = 4.0 * 4.0; // Keep the same 4.0 radius logic

    if dist_to_ray_sq < click_radius_sq {
        // We found a hit. Update result using a spinlock-like atomic CAS for float min
        var current_min_dist = atomicLoad(&result.dist);
        var dist_bits = bitcast<u32>(dist_along_ray);
        
        // Loop until we either successfully write our distance or find a smaller one
        loop {
            if bitcast<f32>(current_min_dist) <= dist_along_ray {
                break;
            }
            let res = atomicCompareExchangeWeak(&result.dist, current_min_dist, dist_bits);
            if res.exchanged {
                atomicStore(&result.id, pid);
                break;
            }
            current_min_dist = res.old_value;
        }
    }
}