#import bevy_pbr::mesh_view_bindings::globals
#import bevy_pbr::forward_io::VertexOutput

// Fonction de pseudo-aléatoire (Hash 2D)
fn hash2d(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Bruit de valeur (Value Noise 2D)
fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Interpolation Hermite douce
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash2d(i);
    let b = hash2d(i + vec2<f32>(1.0, 0.0));
    let c = hash2d(i + vec2<f32>(0.0, 1.0));
    let d = hash2d(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Bruit fractal (Fractal Brownian Motion)
fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 4; i++) {
        value += amplitude * noise2d(p * freq);
        freq *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

@group(3) @binding(0) var grass_texture: texture_2d<f32>;
@group(3) @binding(1) var grass_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    // Les poids de splatting passés par le code Rust
    let weights = mesh.color;
    
    // On utilise la position dans le monde (X, Z) pour échantillonner le bruit
    let pos = mesh.world_position.xz;

    // Génération de quelques couches de bruit
    let n_base = fbm(pos * 0.5);   // Variations larges
    let n_detail = fbm(pos * 2.0); // Variations fines

    // Échantillonnage de la texture d'herbe
    // On répète la texture tous les 4 mètres par exemple
    let grass_uv = pos * 0.25;
    let grass_tex_color = textureSample(grass_texture, grass_sampler, grass_uv).rgb;

    // Définition des couleurs de base, modulées par le bruit
    // On multiplie la couleur de base par une valeur issue du bruit pour casser l'uniformité
    let color_water = vec3<f32>(0.15, 0.45, 0.8) * (0.85 + 0.3 * n_detail);
    let color_plains = grass_tex_color * (0.8 + 0.4 * n_base);
    let color_forest = vec3<f32>(0.2, 0.4, 0.2) * (0.7 + 0.6 * fbm(pos * 1.5));
    let color_desert = vec3<f32>(0.85, 0.75, 0.4) * (0.9 + 0.2 * n_base);

    // Mélange final via les poids de splatting
    let final_color = color_water * weights.r +
                      color_plains * weights.g +
                      color_forest * weights.b +
                      color_desert * weights.a;

    // Output final
    return vec4<f32>(final_color, 1.0);
}
