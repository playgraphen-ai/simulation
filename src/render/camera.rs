//! Orbit / pan camera, plus cursor-to-ground picking for UI tools.

use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::window::PrimaryWindow;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CursorTile>()
            .add_systems(Startup, setup_camera)
            .add_systems(Update, (camera_control, update_cursor_tile));
    }
}

#[derive(Component)]
pub struct MainCamera {
    pub focus: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for MainCamera {
    fn default() -> Self {
        Self { focus: Vec3::new(64., 0., 64.), distance: 80.0, yaw: 0.6, pitch: -0.9 }
    }
}

/// Currently hovered tile coordinates (world grid). `None` when cursor is
/// outside the map or over the UI.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct CursorTile(pub Option<(u32, u32)>);

fn setup_camera(mut commands: Commands) {
    let cam = MainCamera::default();
    let tf = compute_transform(&cam);
    commands.spawn((
        Camera3d::default(),
        tf,
        cam,
        bevy::render::view::NoIndirectDrawing,
        AmbientLight {
            color: Color::srgb(1.0, 0.98, 0.92),
            brightness: 600.0,
            ..default()
        },
    ));
    // Sun with shadows. Shadows work now that we disable MULTI_DRAW_INDIRECT_COUNT
    // at device-init time (see main.rs) — Bevy falls back to per-draw rendering
    // for the shadow pass which the Adreno D3D12 driver handles correctly.
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 80.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn compute_transform(c: &MainCamera) -> Transform {
    let eye = c.focus
        + Vec3::new(
            c.distance * c.yaw.cos() * c.pitch.cos(),
            -c.distance * c.pitch.sin(),
            c.distance * c.yaw.sin() * c.pitch.cos(),
        );
    Transform::from_translation(eye).looking_at(c.focus, Vec3::Y)
}

fn camera_control(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut ev_motion: MessageReader<MouseMotion>,
    mut ev_wheel: MessageReader<MouseWheel>,
    time: Res<Time>,
    mut q: Query<(&mut MainCamera, &mut Transform)>,
) {
    let Ok((mut cam, mut tf)) = q.single_mut() else { return; };
    // Middle mouse or right mouse to rotate.
    if mouse.pressed(MouseButton::Middle) || mouse.pressed(MouseButton::Right) {
        for ev in ev_motion.read() {
            cam.yaw += ev.delta.x * 0.005;
            cam.pitch = (cam.pitch + ev.delta.y * 0.005).clamp(-1.5, -0.1);
        }
    } else {
        ev_motion.clear();
    }
    // Scroll to zoom.
    for ev in ev_wheel.read() {
        cam.distance = (cam.distance - ev.y * 3.0).clamp(15.0, 300.0);
    }
    // WASD to pan along the ground plane, aligned with camera yaw.
    let mut mv = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) { mv.z -= 1.0; }
    if keys.pressed(KeyCode::KeyS) { mv.z += 1.0; }
    if keys.pressed(KeyCode::KeyA) { mv.x -= 1.0; }
    if keys.pressed(KeyCode::KeyD) { mv.x += 1.0; }
    if mv != Vec3::ZERO {
        let speed = cam.distance * 0.5 * time.delta_secs();
        let (s, c) = cam.yaw.sin_cos();
        let forward = Vec3::new(c, 0., s);
        let right = Vec3::new(-s, 0., c);
        cam.focus += (forward * -mv.z + right * mv.x) * speed;
    }
    *tf = compute_transform(&cam);
}

/// Computes the hovered tile by ray-casting the cursor onto the y=0 plane.
fn update_cursor_tile(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    grid: Res<crate::sim::grid::CityGrid>,
    mut cursor: ResMut<CursorTile>,
) {
    cursor.0 = None;
    let Ok(window) = windows.single() else { return; };
    let Some(pos) = window.cursor_position() else { return; };
    let Ok((camera, cam_tf)) = camera_q.single() else { return; };
    let Ok(ray) = camera.viewport_to_world(cam_tf, pos) else { return; };
    // Intersect ray with y=0 plane.
    if ray.direction.y.abs() < 1e-5 { return; }
    let t = -ray.origin.y / ray.direction.y;
    if t <= 0.0 { return; }
    let hit = ray.origin + ray.direction * t;
    let x = hit.x.floor() as i32;
    let z = hit.z.floor() as i32;
    if x >= 0 && z >= 0 && (x as u32) < grid.width && (z as u32) < grid.height {
        cursor.0 = Some((x as u32, z as u32));
    }
}
