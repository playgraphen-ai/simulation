#import bevy_pbr::mesh_view_bindings::view

struct CarMaterialParams {
    people_tex_w: u32,
    roads_tex_w: u32,
    grid_w: u32,
    pad: u32,
}

@group(3) @binding(0) var people_tex: texture_2d<f32>;
@group(3) @binding(1) var roads_tex: texture_2d<f32>;
@group(3) @binding(2) var elevations_tex: texture_2d<f32>;
@group(3) @binding(3) var<uniform> params: CarMaterialParams;
@group(3) @binding(4) var road_points_tex: texture_2d<f32>;

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
    let base = i32(pid * 3u);
    let pw = i32(params.people_tex_w);
    let c0 = vec2<i32>(base % pw, base / pw);
    let c1 = vec2<i32>((base + 1) % pw, (base + 1) / pw);
    let c2 = vec2<i32>((base + 2) % pw, (base + 2) / pw);

    let tex1 = textureLoad(people_tex, c1, 0);
    let tex2 = textureLoad(people_tex, c2, 0);

    let activity = tex1.y;
    let activity_time = tex1.z;
    let current_seg = u32(tex2.x);
    let prev_seg_raw = tex2.y;

    var world_pos = vec3<f32>(0.0, -10000.0, 0.0);
    var rot_matrix = mat3x3<f32>(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );

    if (activity == 0.0 || activity == 4.0) && current_seg != 0xFFFFFFFFu {
        let rw = i32(params.roads_tex_w);
        
        let rbase = i32(current_seg * 4u);
        let rc0 = vec2<i32>(rbase % rw, rbase / rw);
        let rc1 = vec2<i32>((rbase + 1) % rw, (rbase + 1) / rw);
        let rtex0 = textureLoad(roads_tex, rc0, 0);
        let rtex1 = textureLoad(roads_tex, rc1, 0);
        
        let seg_len = max(0.01, rtex1.w); // total length is tex1.w
        let links_offset = u32(rtex1.y);
        let links_count = u32(rtex1.z);
        
        var start_at_b = false;
        if prev_seg_raw >= 1000000.0 {
            start_at_b = true;
        }

        var frac = 0.0;
        if activity_time < 0.0 {
            frac = tex2.z; // start_t
        } else {
            var time_val = max(0.0, activity_time);
            var remaining_frac = clamp(time_val / seg_len, 0.0, 1.0);
            
            if start_at_b {
                // Moving B -> A. Target is A (0.0).
                // Remaining fraction to A is remaining_frac.
                frac = remaining_frac;
            } else {
                // Moving A -> B. Target is B (1.0).
                // Remaining fraction to B is remaining_frac.
                frac = 1.0 - remaining_frac;
            }
        }

        // Interpolation along multi-link segment
        let safe_links_count = max(1u, links_count);
        let target_link_f = frac * f32(safe_links_count - 1u);
        let i = u32(floor(target_link_f));
        let f = fract(target_link_f);
        
        let rw_pts = 1024u;
        let idx0 = links_offset + i;
        let c0 = vec2<i32>(i32(idx0 % rw_pts), i32(idx0 / rw_pts));
        let idx1 = links_offset + min(i + 1u, links_count - 1u);
        let c1 = vec2<i32>(i32(idx1 % rw_pts), i32(idx1 / rw_pts));
        
        let p0 = textureLoad(road_points_tex, c0, 0).xy;
        let p1 = textureLoad(road_points_tex, c1, 0).xy;

        let x = mix(p0.x, p1.x, f) + 0.5;
        let z = mix(p0.y, p1.y, f) + 0.5;

        let el_a = textureLoad(elevations_tex, vec2<i32>(i32(p0.x), i32(p0.y)), 0).x;
        let el_b = textureLoad(elevations_tex, vec2<i32>(i32(p1.x), i32(p1.y)), 0).x;
        let y = mix(el_a, el_b, f) + 0.15;

        world_pos = vec3<f32>(x, y, z);

        var dir = vec3<f32>(p1.x - p0.x, el_b - el_a, p1.y - p0.y);
        if start_at_b {
            dir = -dir;
        }
        
        let len_sq = dot(dir, dir);
        if len_sq > 0.001 {
            let d = normalize(dir);
            let up = vec3<f32>(0.0, 1.0, 0.0);
            let right = normalize(cross(up, d));
            
            // Offset to the right lane
            world_pos = world_pos + right * 0.25;

            let real_up = cross(d, right);
            rot_matrix = mat3x3<f32>(
                right.x, real_up.x, d.x,
                right.y, real_up.y, d.y,
                right.z, real_up.z, d.z
            );
        }
    }

    let local_pos = rot_matrix * vertex.position;
    let final_world = world_pos + local_pos;

    out.world_position = vec4<f32>(final_world, 1.0);
    out.clip_position = view.clip_from_world * out.world_position;

    out.world_normal = rot_matrix * vertex.normal;
    out.uv = vertex.uv;

    if activity == 4.0 {
        out.color = vec3<f32>(1.0, 1.0, 1.0);
    } else {
        let hue = fract(vertex.car_id * 0.6180339887);
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