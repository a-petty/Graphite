use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The classification of a chunk's content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ChunkType {
    Decision,
    Discussion,
    ActionItem,
    StatusUpdate,
    Preference,
    Background,
}

impl ChunkType {
    /// Weight used for PageRank edge weighting.
    /// Higher-value chunk types produce stronger co-occurrence edges.
    pub fn weight(&self) -> f32 {
        match self {
            ChunkType::Decision => 2.0,
            ChunkType::ActionItem => 1.8,
            ChunkType::StatusUpdate => 1.5,
            ChunkType::Preference => 1.5,
            ChunkType::Discussion => 1.0,
            ChunkType::Background => 0.5,
        }
    }
}

/// The memory category, determined by source directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    /// meetings/ — episodic memory
    Episodic,
    /// associates/ — semantic memory
    Semantic,
    /// work/ — procedural memory
    Procedural,
}

/// A chunk of text extracted from a source document, tagged with entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub source_document: String,
    pub section_name: Option<String>,
    pub speaker: Option<String>,
    pub timestamp: Option<i64>,
    pub chunk_type: ChunkType,
    pub memory_category: MemoryCategory,
    pub text: String,
    pub tags: Vec<String>,
    pub created_at: i64,
    #[serde(default)]
    pub projects: Vec<String>,
}

impl Chunk {
    pub fn new(
        source_document: String,
        chunk_type: ChunkType,
        memory_category: MemoryCategory,
        text: String,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            source_document,
            section_name: None,
            speaker: None,
            timestamp: None,
            chunk_type,
            memory_category,
            text,
            tags: Vec::new(),
            created_at: now,
            projects: Vec::new(),
        }
    }
}
