use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// The type of a knowledge entity in the graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Project,
    Technology,
    Organization,
    Location,
    Decision,
    Concept,
    Document,
    Custom(String),
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityType::Person => write!(f, "Person"),
            EntityType::Project => write!(f, "Project"),
            EntityType::Technology => write!(f, "Technology"),
            EntityType::Organization => write!(f, "Organization"),
            EntityType::Location => write!(f, "Location"),
            EntityType::Decision => write!(f, "Decision"),
            EntityType::Concept => write!(f, "Concept"),
            EntityType::Document => write!(f, "Document"),
            EntityType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// A record of a merge operation for audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRecord {
    pub merged_entity_id: String,
    pub merged_entity_name: String,
    pub merged_at: i64,
    pub confidence: f64,
    pub method: String,
}

/// A node in the knowledge graph representing an extracted entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub id: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub entity_type: EntityType,
    pub source_chunks: Vec<String>,
    pub source_documents: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub access_count: u32,
    pub embedding: Option<Vec<f32>>,
    pub rank: f64,
    #[serde(default)]
    pub extraction_confidence: Option<f64>,
    #[serde(default)]
    pub merge_history: Vec<MergeRecord>,
    #[serde(default)]
    pub projects: Vec<String>,
}

impl EntityNode {
    pub fn new(canonical_name: String, entity_type: EntityType) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            canonical_name,
            aliases: Vec::new(),
            entity_type,
            source_chunks: Vec::new(),
            source_documents: Vec::new(),
            created_at: now,
            updated_at: now,
            access_count: 0,
            embedding: None,
            rank: 0.0,
            extraction_confidence: None,
            merge_history: Vec::new(),
            projects: Vec::new(),
        }
    }
}

impl fmt::Display for EntityNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] (rank: {:.3})",
            self.canonical_name, self.entity_type, self.rank
        )
    }
}
