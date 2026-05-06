//! World rendering: camera, ground, simple instanced-ish drawing of tiles,
//! roads, buildings and cars.
//!
//! For the first cut we draw one entity per object (not a full GPU instancer)
//! — good enough to validate the data layer. Switching to truly instanced
//! rendering is the final optimisation step.

pub mod camera;
pub mod world;
pub mod cars;
pub mod terrain;

use bevy::prelude::*;

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            camera::CameraPlugin,
            world::WorldRenderPlugin,
            cars::CarsRenderPlugin,
            terrain::TerrainRenderPlugin,
        ));
    }
}
