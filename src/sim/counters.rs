//! HUD counters, read from CPU shadows each frame.

use bevy::prelude::*;

use super::roads::RoadData;

#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct SimCounters {
    pub people: u32,
    pub cars: u32,
    pub res_occupants: u32,
    pub office_occupants: u32,
    pub shop_occupants: u32,
    pub residential: u32,
    pub offices: u32,
    pub shops: u32,
    pub road_segments: u32,
    pub destroyed_buildings: u32,
}

pub fn update_counters_system(
    roads: Res<RoadData>,
    mut counters: ResMut<SimCounters>,
) {
    counters.road_segments = roads.segments.len() as u32;
    // People and building counters are now updated via GPU readback in apply_gpu_readback.
}
