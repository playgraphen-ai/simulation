//! Spawns new people at the map edge when a `SpawnPeopleRequest` fires.
//!
//! Now fully handled on the GPU. The CPU only reserves space and
//! sends the request to the GPU simulation pipeline.

use bevy::prelude::*;

use crate::sim::people::PeopleData;
use crate::sim::SpawnPeopleRequest;

#[derive(Resource, Default)]
pub struct PendingGpuSpawns {
    pub count: u32,
}

pub fn handle_spawn_requests(
    mut ev: MessageReader<SpawnPeopleRequest>,
    mut people: ResMut<PeopleData>,
    mut pending: ResMut<PendingGpuSpawns>,
) {
    for req in ev.read() {
        let actual_count = if people.len + req.count > crate::sim::people::PEOPLE_CAPACITY {
            crate::sim::people::PEOPLE_CAPACITY.saturating_sub(people.len)
        } else {
            req.count
        };

        if actual_count > 0 {
            // Initialisation locale "fantôme" pour éviter des accès hors limites 
            // ou des stats bizarres sur le CPU avant le readback GPU.
            let start = people.len as usize;
            let end = (people.len + actual_count) as usize;
            for i in start..end {
                people.rows[i] = crate::sim::people::PersonRow::default();
            }

            pending.count += actual_count;
            people.len += actual_count;
            
            // IMPORTANT: On ne met PAS people.dirty = true ici.
            // Si on le faisait, Bevy uploaderait people.rows (rempli de zéros) 
            // vers le GPU, risquant d'écraser le travail du shader de spawn
            // si l'upload système survient après le dispatch.
        }
    }
}
