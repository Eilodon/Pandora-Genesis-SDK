#![allow(unused_variables)]
#![allow(clippy::needless_borrows_for_generic_args)]

use clap::{Parser, Subcommand};
use zenb_core::domain::SessionId;
use zenb_store::EventStore;

#[derive(Parser)]
#[command(name = "zenb-cli")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init { db: String },
    Demo {},
    BeliefDemo {},
    Replay { db: String, session: String },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Init { db } => {
            let key = [3u8; 32];
            let store = EventStore::open(&db, key)?;
            println!("Initialized DB at {}", db);
        }
        Commands::Demo {} => {
            println!("Run the wasm demo with: cargo run -p zenb-wasm-demo");
        }
        Commands::BeliefDemo {} => {
            // quick belief demo: create engine, ingest a few features and print belief
            let mut eng = zenb_core::Engine::new(6.0);
            let ts = 0i64;
            let est = eng.ingest_sensor(&[60.0, 40.0, 6.0, 0.9, 0.1], ts);
            let (dec, changed, policy, deny) = eng.make_control(&est, ts);
            println!("Decision: {:?}, changed: {}", dec, changed);
            if let Some(r) = deny {
                println!("Decision denied: {}", r);
            }
            println!(
                "Belief: mode={:?}, conf={:.3}, p={:?}",
                eng.skandha_pipeline.vedana.mode(),
                eng.skandha_pipeline.vedana.confidence(),
                eng.skandha_pipeline.vedana.probabilities()
            );
            if let Some((m, bits, conf)) = policy {
                println!("PolicyChosen: mode={}, bits={}, conf={}", m, bits, conf);
            }
        }
        Commands::Replay { db, session } => {
            let key = [3u8; 32];
            let store = EventStore::open(&db, key)?;
            let sid_bytes = hex::decode(session)?;
            if sid_bytes.len() != 16 {
                return Err("session id must be 16 bytes hex".into());
            }
            let mut sidarr = [0u8; 16];
            sidarr.copy_from_slice(&sid_bytes);
            let sid = SessionId::new();
            let evs = store.read_events(&sid)?;
            let state = zenb_core::replay::replay_envelopes(&evs)?;
            println!("replay hash: {}", hex::encode(state.hash()));
        }
    }
    Ok(())
}
