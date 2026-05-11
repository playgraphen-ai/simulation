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
    pub bankrupt: u32,
}

pub fn update_counters_system(
    roads: Res<RoadData>,
    buildings: Res<crate::sim::buildings::BuildingData>,
    mut counters: ResMut<SimCounters>,
) {
    counters.road_segments = roads.segments.len() as u32;
    
    let mut res = 0;
    let mut off = 0;
    let mut shop = 0;
    let mut res_occ = 0;
    let mut off_occ = 0;
    let mut shop_occ = 0;

    for b in &buildings.items {
        if b.capacity > 0 {
            match b.btype {
                crate::sim::grid::ZoneType::Residential => {
                    res += 1;
                    res_occ += b.occupants;
                }
                crate::sim::grid::ZoneType::Office => {
                    off += 1;
                    off_occ += b.occupants;
                }
                crate::sim::grid::ZoneType::Shop => {
                    shop += 1;
                    shop_occ += b.occupants;
                }
            }
        }
    }

    counters.residential = res;
    counters.offices = off;
    counters.shops = shop;
    counters.res_occupants = res_occ;
    counters.office_occupants = off_occ;
    counters.shop_occupants = shop_occ;
}
