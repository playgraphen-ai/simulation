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
    pub tax_income_total: u32,
    pub tax_rent_total: u32,
    pub tax_consumption_total: u32,
    pub money_total: u32,
}

pub fn update_counters_system(
    roads: Res<RoadData>,
    buildings: Res<crate::sim::buildings::BuildingData>,
    mut counters: ResMut<SimCounters>,
) {
    if !roads.is_changed() && !buildings.is_changed() { return; }
    
    counters.road_segments = roads.segments.len() as u32;
    
    let mut res = 0;
    let mut off = 0;
    let mut shop = 0;

    for b in &buildings.items {
        if b.capacity > 0 {
            match b.btype {
                crate::sim::grid::ZoneType::Residential => {
                    res += 1;
                }
                crate::sim::grid::ZoneType::Office => {
                    off += 1;
                }
                crate::sim::grid::ZoneType::Shop => {
                    shop += 1;
                }
            }
        }
    }

    counters.residential = res;
    counters.offices = off;
    counters.shops = shop;
}
