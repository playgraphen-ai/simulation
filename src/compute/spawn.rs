//! Spawns new people at the map edge when a `SpawnPeopleRequest` fires.
//!
//! Each new person picks a random residential building as home, a random
//! office as work, and starts in the Travelling state heading to work. Random
//! stats are assigned (money / age).

use bevy::prelude::*;
use rand::prelude::*;

use crate::sim::buildings::BuildingData;
use crate::sim::grid::ZoneType;
use crate::sim::people::{Activity, PeopleData, PersonRow};
use crate::sim::SpawnPeopleRequest;

pub fn handle_spawn_requests(
    mut ev: MessageReader<SpawnPeopleRequest>,
    mut people: ResMut<PeopleData>,
    mut buildings: ResMut<BuildingData>,
) {
    let mut rng = thread_rng();
    for req in ev.read() {
        let homes: Vec<u32> = buildings.items.iter().enumerate()
            .filter(|(_, b)| b.btype == ZoneType::Residential && b.capacity > b.occupants)
            .map(|(i, _)| i as u32)
            .collect();
        let works: Vec<u32> = buildings.items.iter().enumerate()
            .filter(|(_, b)| b.btype == ZoneType::Office && b.capacity > b.occupants)
            .map(|(i, _)| i as u32)
            .collect();
        if homes.is_empty() {
            warn!(
                "Spawn: no residential building available yet. Paint a Residential zone adjacent to a road first."
            );
            continue;
        }
        let has_work = !works.is_empty();
        if !has_work {
            warn!("Spawn: no office yet — people will cycle between home and shopping.");
        }
        for _ in 0..req.count {
            // Re-filter homes that are full due to the loop
            let available_homes: Vec<u32> = homes.iter().copied()
                .filter(|&h| buildings.items[h as usize].capacity > buildings.items[h as usize].occupants)
                .collect();
            if available_homes.is_empty() { break; } // No more space

            let home = *available_homes.choose(&mut rng).unwrap();
            let work = works.choose(&mut rng).copied().unwrap_or(home);
            
            let row = PersonRow {
                money: rng.gen_range(50.0..500.0),
                age: rng.gen_range(18.0..75.0),
                destination: home as f32, // initially home
                home: home as f32,
                work: work as f32,
                activity_code: Activity::Home as u32 as f32,
                activity_time: rng.gen_range(0.5..2.0), // Short wait before leaving
                path_cursor: 0.0,
                current_seg: buildings.items[home as usize].road_seg as f32,
                prev_seg: buildings.items[home as usize].road_seg as f32,
                _pad0: 0.0,
                _pad1: 0.0,
            };
            
            buildings.items[home as usize].occupants += 1;
            people.push(row);
            buildings.dirty = true;
        }
    }
}
