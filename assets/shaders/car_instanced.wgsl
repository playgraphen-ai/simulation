#import bevy_pbr::mesh_view_bindings::view

struct CarMaterialParams {
    people_tex_w: u32,
    roads_tex_w: u32,
    grid_w: u32,
    pad: u32,
}

@group(3) @binding(0) var people_tex: texture_2d<f32>; // Kept for layout compatibility, but unused
@group(3) @binding(1) var roads_tex: texture_2d<f32>; // Unused
@group(3) @binding(2) var elevations_tex: texture_2d<f32>; // Unused
@group(3) @binding(3) var elevations_sampler: sampler; // Unused
@group(3) @binding(4) var<uniform> params: CarMaterialParams; // Unused
@group(3) @binding(5) var road_points_tex: texture_2d<f32>; // Unused
@group(3) @binding(6) var transforms_tex: texture_2d<f32>; // New binding for transforms

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) car_id: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
};

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let pid = u32(vertex.car_id);
    
    // Read from transforms_tex
    let t_w = 1024u;
    let t_idx = pid * 2u;
    
    let tx0 = t_idx % t_w;
    let ty0 = t_idx / t_w;
    let pos_col = textureLoad(transforms_tex, vec2<i32>(i32(tx0), i32(ty0)), 0);
    
    let tx1 = (t_idx + 1u) % t_w;
    let ty1 = (t_idx + 1u) / t_w;
    let rot_col = textureLoad(transforms_tex, vec2<i32>(i32(tx1), i32(ty1)), 0);
    
    let world_pos = pos_col.xyz;
    let is_white = pos_col.w;

    let d_x = rot_col.x;
    let d_z = rot_col.y;
    let r_x = rot_col.z;
    let r_z = rot_col.w;

    // Reconstruct 3x3 rotation matrix (assuming Y-up only rotation)
    let rot_matrix = mat3x3<f32>(
        r_x, 0.0, r_z,
        0.0, 1.0, 0.0,
        d_x, 0.0, d_z
    );

    let local_pos = rot_matrix * vertex.position;
    let final_world = world_pos + local_pos;

    out.world_position = vec4<f32>(final_world, 1.0);
    out.clip_position = view.clip_from_world * out.world_position;

    out.world_normal = rot_matrix * vertex.normal;
    out.uv = vertex.uv;

    if is_white > 0.5 {
        out.color = vec3<f32>(1.0, 1.0, 1.0);
    } else {
        let hue = fract(f32(pid) * 0.6180339887);
        out.color = hsv2rgb(vec3<f32>(hue, 0.8, 0.9));
    }

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ndotl = max(dot(normalize(in.world_normal), light_dir), 0.2);
    let c = in.color * ndotl;
    return vec4<f32>(c, 1.0);
}