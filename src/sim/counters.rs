//! HUD counters, read from CPU shadows each frame.

use bevy::prelude::*;

use super::{buildings::BuildingData, people::PeopleData, roads::RoadData};

#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct SimCounters {
    pub people: u32,
    pub residential: u32,
    pub offices: u32,
    pub shops: u32,
    pub road_segments: u32,
    pub destroyed_buildings: u32,
}

pub fn update_counters_system(
    people: Res<PeopleData>,
    roads: Res<RoadData>,
    buildings: Res<BuildingData>,
    mut counters: ResMut<SimCounters>,
) {
    counters.people = people.len;
    counters.road_segments = roads.segments.len() as u32;
    let (mut r, mut o, mut s) = (0, 0, 0);
    for b in &buildings.items {
        if b.capacity == 0 { continue; }
        match b.btype {
            super::grid::ZoneType::Residential => r += 1,
            super::grid::ZoneType::Office => o += 1,
            super::grid::ZoneType::Shop => s += 1,
        }
    }
    counters.residential = r;
    counters.offices = o;
    counters.shops = s;
}
