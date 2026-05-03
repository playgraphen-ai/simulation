#![allow(dead_code)]

use bevy::prelude::*;

use crate::sim::roads::RoadData;

#[derive(Resource, Default)]
pub struct SimTiming {
    pub repath_acc: f32,
}

pub fn cpu_sim_tick(
    time: Res<Time>,
    mut roads: ResMut<RoadData>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 { return; }

    // On smoothed return speed_mean to 1.0 since traffic computation is now offloaded or simplified
    for seg in roads.segments.iter_mut() {
        seg.speed_mean = seg.speed_mean * 0.95 + 1.0 * 0.05;
    }
    
    if !roads.segments.is_empty() {
        roads.refresh_all_rows();
    }
}