use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use std::fs::OpenOptions;
use std::io::Write;
use serde::Serialize;
use crate::sim::counters::SimCounters;
use crate::sim::people::PeopleData;
use crate::sim::buildings::BuildingData;
use std::panic;
use std::sync::Once;

static INIT_PANIC_HOOK: Once = Once::new();

#[derive(Serialize)]
struct TelemetryLog {
    timestamp: f64,
    fps: f64,
    population: u32,
    cars_on_road: u32,
    buildings_total: u32,
    buildings_abandoned: u32,
    destroyed_buildings: u32,
}

pub fn json_logger_system(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    counters: Res<SimCounters>,
    people: Res<PeopleData>,
    buildings: Res<BuildingData>,
    mut acc: Local<f32>,
    mut init: Local<bool>,
) {
    if !*init {
        // Create or overwrite the log file at launch
        if let Ok(mut file) = OpenOptions::new().write(true).create(true).truncate(true).open("log.json") {
            let _ = writeln!(file, "[\n  {{\"event\": \"game_started\"}}");
        }
        
        INIT_PANIC_HOOK.call_once(|| {
            let default_hook = panic::take_hook();
            panic::set_hook(Box::new(move |panic_info| {
                let payload = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
                    s.to_string()
                } else {
                    "Unknown panic".to_string()
                };
                
                let location = if let Some(loc) = panic_info.location() {
                    format!("{}:{}:{}", loc.file(), loc.line(), loc.column())
                } else {
                    "unknown location".to_string()
                };

                let trace = format!("Panic at {}: {}", location, payload);
                if let Ok(mut file) = OpenOptions::new().append(true).open("log.json") {
                    let _ = writeln!(file, ",\n  {{\"event\": \"panic\", \"trace\": {:?}}}", trace);
                }
                
                default_hook(panic_info);
            }));
        });
        
        *init = true;
    }

    *acc += time.delta_secs();
    if *acc >= 1.0 {
        *acc -= 1.0;
        
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.smoothed())
            .unwrap_or(0.0);
            
        let mut abandoned = 0;
        for b in &buildings.items {
            if b.capacity == 0 {
                abandoned += 1;
            }
        }
        
        let mut cars_on_road = 0;
        for i in 0..people.len as usize {
            if people.rows[i].activity_code == 0.0 {
                cars_on_road += 1;
            }
        }

        let log = TelemetryLog {
            timestamp: time.elapsed_secs_f64(),
            fps,
            population: people.len,
            cars_on_road,
            buildings_total: buildings.items.len() as u32,
            buildings_abandoned: abandoned,
            destroyed_buildings: counters.destroyed_buildings,
        };

        if let Ok(json) = serde_json::to_string(&log) {
            if let Ok(mut file) = OpenOptions::new().append(true).open("log.json") {
                let _ = writeln!(file, ",\n  {}", json);
            }
        }
    }
}