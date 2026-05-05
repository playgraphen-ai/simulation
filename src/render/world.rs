//! Draws the ground, zoned tiles, roads and buildings.
//!
//! Buildings use GLTF scenes from the Kenney City Kit (CC0). Roads and zone
//! decals stay as simple plane meshes for now.

use bevy::prelude::*;

use crate::sim::{
    buildings::{BuildingData, MAX_LEVEL},
    grid::{CityGrid, Tile, ZoneType},
    roads::RoadData,
};

pub struct WorldRenderPlugin;

impl Plugin for WorldRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldVisuals>()
            .add_systems(Startup, (setup_ground.after(crate::sim::startup), build_palette))
            .add_systems(Update, (sync_zone_view, sync_road_view, sync_building_view));
    }
}

#[derive(Resource, Default)]
pub struct WorldVisuals {
    pub tile_mesh: Handle<Mesh>,
    pub road_mesh: Handle<Mesh>,
    pub mat_zone_res: Handle<StandardMaterial>,
    pub mat_zone_off: Handle<StandardMaterial>,
    pub mat_zone_shop: Handle<StandardMaterial>,
    pub mat_road: Handle<StandardMaterial>,
    /// Per (btype, level) GLTF scene handle.
    pub residential_scenes: Vec<Handle<Scene>>, // 5
    pub office_scenes: Vec<Handle<Scene>>,      // 5
    pub shop_scenes: Vec<Handle<Scene>>,        // 5
    pub road_scene: Handle<Scene>,
    pub car_scene: Handle<Scene>,
}

#[derive(Component)]
struct ZoneMarker;

#[derive(Component)]
struct RoadMarker(pub u32);

#[derive(Component)]
pub struct BuildingMarker(pub u32, pub u32); // (id, level)

use bevy::render::render_resource::PrimitiveTopology;
use bevy::mesh::Indices;
use bevy::asset::RenderAssetUsages;
use crate::sim::grid::Biome;

fn setup_ground(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    grid: Res<CityGrid>,
) {
    let w = grid.width;
    let h = grid.height;
    
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((w * h) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((w * h) as usize);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity((w * h) as usize);
    let mut indices: Vec<u32> = Vec::with_capacity(((w - 1) * (h - 1) * 6) as usize);
    
    // Generate vertices
    for y in 0..h {
        for x in 0..w {
            let i = grid.idx(x, y);
            let elev = grid.elevations[i];
            // Center the grid on the tile visually
            positions.push([x as f32 + 0.5, elev, y as f32 + 0.5]);
            
            let color = match grid.biomes[i] {
                Biome::Water => [0.15, 0.45, 0.8, 1.0],
                Biome::Plains => [0.4, 0.6, 0.3, 1.0],
                Biome::Forest => [0.2, 0.4, 0.2, 1.0],
                Biome::Desert => [0.85, 0.75, 0.4, 1.0],
            };
            colors.push(color);
            normals.push([0.0, 1.0, 0.0]); // Simple upward normals for flat shading look
        }
    }
    
    // Compute basic normals
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            let top_left = y * w + x;
            let top_right = top_left + 1;
            let bottom_left = (y + 1) * w + x;
            let bottom_right = bottom_left + 1;
            
            // Triangle 1
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);
            
            // Triangle 2
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }
    
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_normals(); // Let Bevy compute smooth normals
    
    let mat = materials.add(StandardMaterial {
        perceptual_roughness: 0.95,
        ..default()
    });
    
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(mat),
        Transform::default(),
    ));
}

fn build_palette(
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut vis: ResMut<WorldVisuals>,
) {
    vis.tile_mesh = meshes.add(Plane3d::default().mesh().size(0.96, 0.96));
    vis.road_mesh = meshes.add(Plane3d::default().mesh().size(0.96, 0.96));
    vis.mat_zone_res = materials.add(tint(0.2, 0.7, 0.25, 0.4));
    vis.mat_zone_off = materials.add(tint(0.25, 0.45, 0.85, 0.4));
    vis.mat_zone_shop = materials.add(tint(0.9, 0.65, 0.2, 0.4));
    vis.mat_road = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.15, 0.18),
        perceptual_roughness: 0.9,
        ..default()
    });
    vis.residential_scenes = (0..=MAX_LEVEL)
        .map(|lvl| asset_server.load(GltfAssetLabel::Scene(0)
            .from_asset(format!("models/buildings/residential_{}.glb", lvl))))
        .collect();
    vis.office_scenes = (0..=MAX_LEVEL)
        .map(|lvl| asset_server.load(GltfAssetLabel::Scene(0)
            .from_asset(format!("models/buildings/office_{}.glb", lvl))))
        .collect();
    vis.shop_scenes = (0..=MAX_LEVEL)
        .map(|lvl| asset_server.load(GltfAssetLabel::Scene(0)
            .from_asset(format!("models/buildings/shop_{}.glb", lvl))))
        .collect();
    vis.road_scene = asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/roads/road_straight.glb"));
    vis.car_scene = asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/vehicles/sedan.glb"));
}

fn tint(r: f32, g: f32, b: f32, a: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgba(r, g, b, a),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    }
}

fn sync_zone_view(
    mut commands: Commands,
    grid: Res<CityGrid>,
    vis: Res<WorldVisuals>,
    existing: Query<Entity, With<ZoneMarker>>,
) {
    if !grid.is_changed() { return; }
    for e in &existing {
        commands.entity(e).despawn();
    }
    for y in 0..grid.height {
        for x in 0..grid.width {
            if let Some(Tile::Zone(z)) = grid.get(x, y) {
                let mat = match z {
                    ZoneType::Residential => vis.mat_zone_res.clone(),
                    ZoneType::Office => vis.mat_zone_off.clone(),
                    ZoneType::Shop => vis.mat_zone_shop.clone(),
                };
                let elev = grid.elevations[grid.idx(x, y)];
                commands.spawn((
                    Mesh3d(vis.tile_mesh.clone()),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(x as f32 + 0.5, elev + 0.02, y as f32 + 0.5),
                    ZoneMarker,
                ));
            }
        }
    }
}

fn sync_road_view(
    mut commands: Commands,
    roads: Res<RoadData>,
    grid: Res<CityGrid>,
    vis: Res<WorldVisuals>,
    existing: Query<(Entity, &RoadMarker)>,
) {
    if !roads.is_changed() { return; }
    let n = roads.segments.len() as u32;
    let mut existing_ids = std::collections::HashSet::new();
    for (e, m) in &existing {
        if m.0 >= n {
            commands.entity(e).despawn();
        } else {
            existing_ids.insert(m.0);
        }
    }
    for (id, seg) in roads.segments.iter().enumerate() {
        let id_u32 = id as u32;
        if existing_ids.contains(&id_u32) { continue; }
        let x = (seg.a.0 + seg.b.0) as f32 * 0.5 + 0.5;
        let z = (seg.a.1 + seg.b.1) as f32 * 0.5 + 0.5;
        // Rotate so the straight road segment aligns with the A—B axis.
        let dx = seg.b.0 as f32 - seg.a.0 as f32;
        let dz = seg.b.1 as f32 - seg.a.1 as f32;
        let angle = dz.atan2(dx);
        let len = (dx * dx + dz * dz).sqrt();

        let elev_a = grid.elevations[grid.idx(seg.a.0, seg.a.1)];
        let elev_b = grid.elevations[grid.idx(seg.b.0, seg.b.1)];
        let y = (elev_a + elev_b) * 0.5 + 0.01;

        let mut tf = Transform::from_xyz(x, y, z).with_rotation(Quat::from_rotation_y(-angle));
        tf.scale.x = len;
        tf.scale.z = 2.0; // 2 tiles wide (visually)

        // Pitch road if there is an elevation difference
        let pitch = (elev_b - elev_a).atan2(len);
        tf.rotate_local_z(-pitch);

        commands.spawn((
            SceneRoot(vis.road_scene.clone()),
            tf,
            RoadMarker(id_u32),
        ));
    }
}

fn sync_building_view(
    mut commands: Commands,
    buildings: Res<BuildingData>,
    grid: Res<CityGrid>,
    vis: Res<WorldVisuals>,
    mut existing: Query<(Entity, &mut BuildingMarker, &mut SceneRoot, &mut Transform)>,
) {
    if !buildings.is_changed() { return; }
    let mut seen = std::collections::HashSet::new();
    for (e, mut marker, mut scene, mut _tf) in &mut existing {
        let id = marker.0 as usize;
        if id >= buildings.items.len() {
            commands.entity(e).despawn();
            continue;
        }
        let b = &buildings.items[id];
        if b.capacity == 0 {
            commands.entity(e).despawn();
            continue;
        }
        if marker.1 != b.level {
            scene.0 = scene_for(&vis, b.btype, b.level);
            marker.1 = b.level;
        }
        seen.insert(marker.0);
    }
    for (id, b) in buildings.items.iter().enumerate() {
        let id_u32 = id as u32;
        if seen.contains(&id_u32) || b.capacity == 0 { continue; }
        let elev = grid.elevations[grid.idx(b.tile.0, b.tile.1)];
        commands.spawn((
            SceneRoot(scene_for(&vis, b.btype, b.level)),
            Transform::from_xyz(b.tile.0 as f32 + 0.5, elev, b.tile.1 as f32 + 0.5),
            BuildingMarker(id_u32, b.level),
        ));
    }
}

fn scene_for(vis: &WorldVisuals, bt: ZoneType, lvl: u32) -> Handle<Scene> {
    let lvl = (lvl as usize).min(MAX_LEVEL as usize);
    match bt {
        ZoneType::Residential => vis.residential_scenes[lvl].clone(),
        ZoneType::Office => vis.office_scenes[lvl].clone(),
        ZoneType::Shop => vis.shop_scenes[lvl].clone(),
    }
}
