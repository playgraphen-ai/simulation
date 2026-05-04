#![allow(dead_code)]

use bevy::prelude::*;

use crate::sim::roads::RoadData;

#[derive(Resource, Default)]
pub struct SimTiming {
    pub repath_acc: f32,
}

pub fn cpu_sim_tick(
    time: Res<Time>,
    _roads: ResMut<RoadData>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 { return; }

    // Road speed and traffic are now fully simulated on the GPU using probabilistic estimation.
    // The CPU no longer needs to run fallback smoothing for roads.
}