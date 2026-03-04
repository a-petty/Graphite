use serde::{Deserialize, Serialize};

use crate::chunk::{ChunkType, MemoryCategory};

/// An edge in the knowledge graph representing co-occurrence of two entities
/// within the same chunk of text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoOccurrenceEdge {
    pub chunk_id: String,
    pub chunk_type: ChunkType,
    pub memory_category: MemoryCategory,
    pub timestamp: Option<i64>,
    pub source_document: String,
    pub weight: f32,
}

impl CoOccurrenceEdge {
    pub fn new(
        chunk_id: String,
        chunk_type: ChunkType,
        memory_category: MemoryCategory,
        timestamp: Option<i64>,
        source_document: String,
    ) -> Self {
        let weight = chunk_type.weight();
        Self {
            chunk_id,
            chunk_type,
            memory_category,
            timestamp,
            source_document,
            weight,
        }
    }

    /// Returns the edge strength as f64, matching the EdgeKind::strength() pattern
    /// used by PageRank.
    pub fn strength(&self) -> f64 {
        self.weight as f64
    }
}
