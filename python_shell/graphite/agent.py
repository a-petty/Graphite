"""Graphite Agent — Knowledge graph query agent.

Routes user queries through the Graphite knowledge graph using
MemoryContextManager for three-tier Anchor & Expand context assembly,
and provides knowledge graph tools for deeper exploration.
"""

import json
import logging
import re
import ast
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown

# Rust Core Import
from graphite.semantic_engine import PyKnowledgeGraph

# Local Imports
from .embeddings import EmbeddingManager
from .context import MemoryContextManager
from .llm import StubClient, OllamaClient, MLXClient, OpenAIClient, AnthropicClient
from .config import GraphiteConfig

# Setup Rich Console & Logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, show_path=False)]
)
log = logging.getLogger("graphite")

SYSTEM_PROMPT = """You are Graphite, a knowledge graph memory assistant.

You have access to a knowledge graph built from ingested documents (meetings, \
profiles, work records). The graph contains entities (people, projects, \
technologies, organizations, decisions, concepts) connected by co-occurrence \
edges — entities that appear together in the same text chunks.

## Context Structure

You receive three tiers of knowledge context:

1. **Key Entities** — The most relevant entities with their types, importance \
scores, and connections
2. **Evidence** — Actual text chunks from documents where relevant entities co-occur
3. **Peripheral Entities** — Brief summaries of related but less central entities

The evidence tier is the most important — it contains the actual source text. \
Use it to ground your answers in real information rather than speculation.

## Available Tools

You can call tools to explore the knowledge graph further:

- `graph_status()` — Get graph statistics (entity count, edges, chunks)
- `entity_profile(name)` — Full profile: type, aliases, co-occurrences, recent mentions
- `entity_mentions(name)` — Text chunks where an entity is mentioned
- `cooccurrences(name)` — Entities that co-occur with a given entity
- `evidence(name_a, name_b)` — Text chunks where two entities co-occur together
- `timeline(name)` — Chronological timeline of mentions for an entity
- `search_entities(query)` — Semantic search for entities matching a description
- `knowledge_map()` — PageRank-ranked overview of the whole graph

## Response Format

Always respond using <think> and <action> tags:

<think>
1. What entities and evidence are in my context?
2. Do I have enough information to answer, or do I need to look up more?
3. What is my answer based on the evidence?
</think>

To call a tool:
<action>tool_name('argument')</action>

If you have enough context to answer without tools, provide your answer as \
plain text after your <think> block. Do NOT wrap your final answer in <action> tags.

## Guidelines

- Ground answers in evidence chunks when available — cite specific text
- Say "I don't have information about that" rather than speculating
- If the knowledge graph is empty or has no relevant entities, say so clearly
- When entities have multiple mentions, look for patterns across time
"""


def _format_timestamp(ts) -> Optional[str]:
    """Format a Unix timestamp to YYYY-MM-DD, or None."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return None


class GraphiteAgent:
    """Knowledge graph query agent for Graphite.

    Bridges the Rust knowledge graph with LLM-powered query answering.
    Uses MemoryContextManager for three-tier context assembly and provides
    knowledge graph tools for deeper exploration.
    """

    KNOWN_TOOLS = {
        'graph_status', 'entity_profile', 'entity_mentions',
        'cooccurrences', 'evidence', 'timeline',
        'search_entities', 'knowledge_map',
    }

    def __init__(self, project_root: Path, provider: str = "stub", model_name: str = "llama3.1:8b"):
        self.project_root = project_root.resolve()
        self.model_name = model_name

        # Knowledge graph — loaded in initialize()
        self.kg: Optional[PyKnowledgeGraph] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.context_manager: Optional[MemoryContextManager] = None
        self.config: Optional[GraphiteConfig] = None

        # LLM client
        if provider == "ollama":
            self.llm = OllamaClient(model=model_name)
        elif provider == "mlx":
            self.llm = MLXClient(model=model_name)
        elif provider == "openai":
            self.llm = OpenAIClient(model=model_name)
        elif provider == "anthropic":
            self.llm = AnthropicClient(model=model_name)
        else:
            self.llm = StubClient()

        # Conversation state for multi-turn chat
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_tokens: int = 50000

    def initialize(self):
        """Load the knowledge graph and prepare for queries."""
        console.print(Panel(
            f"[bold blue]Graphite Agent[/bold blue]\nTarget: {self.project_root}",
            border_style="blue"
        ))

        # Load config
        toml_path = self.project_root / ".graphite.toml"
        if toml_path.exists():
            try:
                self.config = GraphiteConfig.from_toml(toml_path)
            except Exception:
                self.config = GraphiteConfig()
        else:
            self.config = GraphiteConfig()

        # Graph storage lives at config.graph_root (default ~) — independent
        # of project_root, which is only used for tagging / TOML lookup.
        # All access goes through the graphited daemon; the agent never
        # touches the msgpack file directly.
        self.graph_root = Path(self.config.graph_root).expanduser().resolve()
        from graphite.client import DaemonBackedGraph, DaemonUnavailable, GraphiteClient

        self._daemon_client = GraphiteClient()
        try:
            self._daemon_client.ping()
        except DaemonUnavailable as e:
            raise RuntimeError(
                f"graphited is not running: {e}. "
                f"Start it with `graphited` before invoking the agent."
            )
        self.kg = DaemonBackedGraph(self._daemon_client)
        log.info(f"Connected to graphited (graph root: {self.graph_root})")

        # Initialize embeddings and context manager
        self.embedding_manager = EmbeddingManager()
        self.context_manager = MemoryContextManager(
            knowledge_graph=self.kg,
            embedding_manager=self.embedding_manager,
            config=self.config,
            model=self.model_name,
        )

        # Display statistics
        stats = json.loads(self.kg.get_statistics())
        stats_tree = Tree("[bold]Knowledge Graph[/bold]")
        stats_tree.add(f"[cyan]Entities:[/cyan] {stats.get('entity_count', 0)}")
        stats_tree.add(f"[yellow]Co-occurrence edges:[/yellow] {stats.get('edge_count', 0)}")
        stats_tree.add(f"[green]Chunks stored:[/green] {stats.get('chunk_count', 0)}")
        stats_tree.add(f"[magenta]Documents indexed:[/magenta] {stats.get('documents_indexed', 0)}")

        by_type = stats.get("entities_by_type", {})
        if by_type:
            type_branch = stats_tree.add("[dim]By type:[/dim]")
            for etype, count in sorted(by_type.items()):
                type_branch.add(f"{etype}: {count}")

        console.print(stats_tree)
        console.print("[bold green]\u2713[/bold green] Graphite Agent ready.\n")

    # ── Entity Resolution ──

    def _resolve_entity(self, ref: str) -> dict:
        """Resolve an entity reference (ID or name) to an entity dict."""
        entity_json = self.kg.get_entity(ref)
        if entity_json is not None:
            return json.loads(entity_json)

        results_json = self.kg.search_entities(ref, 5)
        results = json.loads(results_json)
        if not results:
            raise ValueError(f"Entity not found: '{ref}'")

        ref_lower = ref.lower()
        for entity in results:
            if entity["canonical_name"].lower() == ref_lower:
                return entity

        return results[0]

    # ── Knowledge Graph Tools ──

    def graph_status(self) -> Dict:
        """Get knowledge graph statistics."""
        stats = json.loads(self.kg.get_statistics())
        lines = [
            "Knowledge Graph Status:",
            f"  Entities: {stats.get('entity_count', 0)}",
            f"  Co-occurrence edges: {stats.get('edge_count', 0)}",
            f"  Chunks stored: {stats.get('chunk_count', 0)}",
            f"  Documents indexed: {stats.get('documents_indexed', 0)}",
        ]
        by_type = stats.get("entities_by_type", {})
        if by_type:
            lines.append("  Entities by type:")
            for etype, count in sorted(by_type.items()):
                lines.append(f"    {etype}: {count}")
        return {"success": True, "content": "\n".join(lines)}

    def entity_profile(self, name: str) -> Dict:
        """Get full profile for an entity."""
        try:
            ent = self._resolve_entity(name)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        entity_id = ent["id"]
        lines = [f"# {ent['canonical_name']} ({ent.get('entity_type', 'Unknown')})"]

        aliases = ent.get("aliases", [])
        if aliases:
            lines.append(f"**Aliases:** {', '.join(aliases)}")

        docs = ent.get("source_documents", [])
        if docs:
            lines.append(f"**Source documents:** {', '.join(docs)}")

        # Co-occurrences
        cooc_json = self.kg.get_cooccurrences(entity_id)
        coocs = json.loads(cooc_json)
        if coocs:
            neighbor_counts = {}
            for item in coocs:
                nid = item[0]
                neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1

            sorted_neighbors = sorted(
                neighbor_counts.items(), key=lambda x: x[1], reverse=True
            )
            lines.append("\n**Co-occurs with:**")
            for nid, count in sorted_neighbors[:10]:
                n_json = self.kg.get_entity(nid)
                if n_json:
                    n = json.loads(n_json)
                    lines.append(
                        f"  - {n['canonical_name']} ({n.get('entity_type', '?')}): {count}x"
                    )

        # Recent chunks
        chain_json = self.kg.get_temporal_chain(entity_id)
        chunks = json.loads(chain_json)
        if chunks:
            lines.append(f"\n**Recent mentions** ({len(chunks)} total):")
            for chunk in chunks[:5]:
                ts_str = _format_timestamp(chunk.get("timestamp"))
                ctype = chunk.get("chunk_type", "?")
                text = chunk.get("text", "").strip()
                if len(text) > 200:
                    text = text[:197] + "..."
                lines.append(f"  [{ctype}] ({ts_str}) {text}")

        return {"success": True, "content": "\n".join(lines)}

    def entity_mentions(self, name: str, limit: int = 20) -> Dict:
        """Get chunks where an entity is mentioned."""
        try:
            ent = self._resolve_entity(name)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        entity_id = ent["id"]
        chain_json = self.kg.get_temporal_chain(entity_id)
        chunks = json.loads(chain_json)

        if not chunks:
            return {
                "success": True,
                "content": f"No mentions found for '{ent['canonical_name']}'.",
            }

        chunks = chunks[:limit]
        lines = [f"Mentions of **{ent['canonical_name']}** ({len(chunks)} chunk(s)):"]
        for chunk in chunks:
            ts_str = _format_timestamp(chunk.get("timestamp"))
            ctype = chunk.get("chunk_type", "Unknown")
            source = chunk.get("source_document", "unknown")
            text = chunk.get("text", "").strip()
            if len(text) > 300:
                text = text[:297] + "..."
            lines.append(f"\n**[{ctype}]** {source} ({ts_str})")
            lines.append(f"> {text}")

        return {"success": True, "content": "\n".join(lines)}

    def cooccurrences(self, name: str) -> Dict:
        """Get entities that co-occur with a given entity."""
        try:
            ent = self._resolve_entity(name)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        entity_id = ent["id"]
        cooc_json = self.kg.get_cooccurrences(entity_id)
        coocs = json.loads(cooc_json)

        if not coocs:
            return {
                "success": True,
                "content": f"No co-occurrences found for '{ent['canonical_name']}'.",
            }

        neighbor_data = {}
        for item in coocs:
            nid = item[0]
            edge = item[1]
            if nid not in neighbor_data:
                neighbor_data[nid] = {"count": 0, "latest_ts": None}
            neighbor_data[nid]["count"] += 1
            ts = edge.get("timestamp")
            if ts is not None:
                prev = neighbor_data[nid]["latest_ts"]
                if prev is None or ts > prev:
                    neighbor_data[nid]["latest_ts"] = ts

        entries = []
        for nid, data in neighbor_data.items():
            n_json = self.kg.get_entity(nid)
            if n_json is None:
                continue
            n = json.loads(n_json)
            entries.append({
                "name": n["canonical_name"],
                "type": n.get("entity_type", "Unknown"),
                "count": data["count"],
                "latest_ts": data["latest_ts"],
            })

        entries.sort(key=lambda x: x["count"], reverse=True)

        lines = [
            f"Co-occurrences for **{ent['canonical_name']}** "
            f"({ent.get('entity_type', 'Unknown')}):"
        ]
        for e in entries:
            ts_str = _format_timestamp(e["latest_ts"])
            lines.append(
                f"  - {e['name']} ({e['type']}): "
                f"{e['count']} co-occurrence(s), last: {ts_str}"
            )

        return {"success": True, "content": "\n".join(lines)}

    def evidence(self, name_a: str, name_b: str, limit: int = 15) -> Dict:
        """Get chunks where two entities co-occur."""
        try:
            ent_a = self._resolve_entity(name_a)
            ent_b = self._resolve_entity(name_b)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        id_a = ent_a["id"]
        id_b = ent_b["id"]

        chunks_json = self.kg.get_chunks_for_entities(json.dumps([id_a, id_b]))
        all_chunks = json.loads(chunks_json)

        both = []
        for chunk in all_chunks:
            tags = set(chunk.get("tags", []))
            if id_a in tags and id_b in tags:
                both.append(chunk)

        if not both:
            return {
                "success": True,
                "content": (
                    f"No shared chunks between "
                    f"'{ent_a['canonical_name']}' and '{ent_b['canonical_name']}'."
                ),
            }

        both.sort(key=lambda c: c.get("timestamp") or 0, reverse=True)
        both = both[:limit]

        lines = [
            f"Evidence for **{ent_a['canonical_name']}** <> "
            f"**{ent_b['canonical_name']}** ({len(both)} chunk(s)):"
        ]
        for chunk in both:
            ts_str = _format_timestamp(chunk.get("timestamp"))
            ctype = chunk.get("chunk_type", "?")
            source = chunk.get("source_document", "unknown")
            text = chunk.get("text", "").strip()
            if len(text) > 400:
                text = text[:397] + "..."
            lines.append(f"\n**[{ctype}]** {source} ({ts_str})")
            lines.append(f"> {text}")

        return {"success": True, "content": "\n".join(lines)}

    def timeline(self, name: str, limit: int = 30) -> Dict:
        """Get chronological timeline of mentions for an entity."""
        try:
            ent = self._resolve_entity(name)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        entity_id = ent["id"]
        chain_json = self.kg.get_temporal_chain(entity_id)
        chunks = json.loads(chain_json)

        if not chunks:
            return {
                "success": True,
                "content": f"No timeline data for '{ent['canonical_name']}'.",
            }

        # Oldest-first for timeline view
        chunks = list(reversed(chunks[:limit]))

        lines = [f"Timeline for **{ent['canonical_name']}** ({len(chunks)} entries):"]
        for chunk in chunks:
            ts_str = _format_timestamp(chunk.get("timestamp"))
            ctype = chunk.get("chunk_type", "?")
            source = chunk.get("source_document", "unknown")
            text = chunk.get("text", "").strip()
            if len(text) > 200:
                text = text[:197] + "..."
            lines.append(f"\n[{ts_str}] **{ctype}** - {source}")
            lines.append(f"> {text}")

        return {"success": True, "content": "\n".join(lines)}

    def search_entities(self, query: str, top_n: int = 10) -> Dict:
        """Semantic search for entities matching a query."""
        stats = json.loads(self.kg.get_statistics())
        if stats.get("entity_count", 0) == 0:
            return {
                "success": True,
                "content": "Knowledge graph is empty — no entities to search.",
            }

        pagerank_json = self.kg.compute_pagerank()
        ranked = json.loads(pagerank_json)
        all_ids = [eid for eid, _ in ranked]

        # Ensure entities are embedded
        entities_for_embed = []
        for eid in all_ids:
            ej = self.kg.get_entity(eid)
            if ej:
                entities_for_embed.append(json.loads(ej))
        self.embedding_manager.embed_entities(entities_for_embed, self.kg)

        scored = self.embedding_manager.find_relevant_entities_scored(
            query, all_ids, top_n=top_n * 3
        )

        if not scored:
            return {
                "success": True,
                "content": f"No relevant entities found for: {query}",
            }

        # Re-rank with PageRank
        pr_map = {eid: score for eid, score in ranked}
        max_pr = max(pr_map.values()) if pr_map else 1.0
        sim_weight = self.config.similarity_weight if self.config else 0.80
        pr_weight = self.config.pagerank_weight if self.config else 0.20

        reranked = []
        for eid, sim in scored:
            pr = pr_map.get(eid, 0.0)
            normalized_pr = pr / max_pr if max_pr > 0 else 0.0
            combined = sim_weight * sim + pr_weight * normalized_pr
            reranked.append((eid, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked = reranked[:top_n]

        lines = [f"Entities relevant to '{query}':"]
        for i, (eid, score) in enumerate(reranked, 1):
            ej = self.kg.get_entity(eid)
            if ej:
                e = json.loads(ej)
                lines.append(
                    f"  {i}. **{e['canonical_name']}** "
                    f"({e.get('entity_type', '?')}) — score: {score:.3f}"
                )

        return {"success": True, "content": "\n".join(lines)}

    def knowledge_map(self, max_entities: int = 50) -> Dict:
        """Get PageRank-ranked knowledge map."""
        pagerank_json = self.kg.compute_pagerank()
        ranked = json.loads(pagerank_json)

        if not ranked:
            return {
                "success": True,
                "content": "Knowledge graph is empty — no entities indexed.",
            }

        ranked = ranked[:max_entities]

        by_type = {}
        for entity_id, score in ranked:
            entity_json = self.kg.get_entity(entity_id)
            if entity_json is None:
                continue
            entity = json.loads(entity_json)
            etype = entity.get("entity_type", "Unknown")
            if etype not in by_type:
                by_type[etype] = []

            cooc_json = self.kg.get_cooccurrences(entity_id)
            coocs = json.loads(cooc_json)
            neighbor_names = set()
            for item in coocs:
                nid = item[0]
                n_json = self.kg.get_entity(nid)
                if n_json:
                    n = json.loads(n_json)
                    neighbor_names.add(n["canonical_name"])

            by_type[etype].append({
                "name": entity["canonical_name"],
                "score": score,
                "neighbors": sorted(neighbor_names)[:5],
            })

        lines = ["Knowledge Map (PageRank-ranked):"]
        for etype, entities in sorted(by_type.items()):
            lines.append(f"\n## {etype}")
            for e in entities:
                neighbors_str = (
                    ", ".join(e["neighbors"]) if e["neighbors"] else "none"
                )
                lines.append(
                    f"  - **{e['name']}** (score: {e['score']:.4f}) "
                    f"-> co-occurs with: {neighbors_str}"
                )

        return {"success": True, "content": "\n".join(lines)}

    # ── Response Parsing ──

    def _parse_response(self, response: str) -> Tuple[Optional[str], List[Dict], str]:
        """Parse LLM response to extract <think>, <action> blocks, and plain text."""
        thoughts = None
        actions = []

        # Normalize whitespace in tags
        normalized = re.sub(
            r'<\s*(/?)\s*(think|action)\s*>', r'<\1\2>', response
        )

        think_match = re.search(
            r'<think>(.*?)(?:</think>|$)', normalized, re.DOTALL
        )
        if think_match:
            thoughts = think_match.group(1).strip()

        action_matches = re.findall(
            r'<action>(.*?)(?:</action>|$)', normalized, re.DOTALL
        )
        for action_text in action_matches:
            parsed = self._parse_single_action(action_text.strip())
            if parsed:
                actions.append(parsed)

        if not actions:
            actions = self._fallback_parse_actions(normalized)

        # Extract plain text (everything outside tags)
        plain_text = re.sub(
            r'<think>.*?(?:</think>|$)', '', normalized, flags=re.DOTALL
        )
        plain_text = re.sub(
            r'<action>.*?(?:</action>|$)', '', plain_text, flags=re.DOTALL
        )
        plain_text = plain_text.strip()

        if not thoughts and not actions and not plain_text:
            log.warning(
                f"Could not parse any content from LLM response: {response[:500]}"
            )

        return thoughts, actions, plain_text

    def _parse_single_action(self, action_text: str) -> Optional[Dict]:
        """Parse a single tool_name(args) pattern."""
        match = re.match(r'(\w+)\((.*)\)', action_text, re.DOTALL)
        if not match:
            log.warning(
                f"Could not match tool call pattern: {action_text[:100]}"
            )
            return None

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        if not args_str:
            return {'tool': tool_name, 'args': []}

        try:
            args = ast.literal_eval(f"[{args_str}]")
            return {'tool': tool_name, 'args': args}
        except (ValueError, SyntaxError):
            pass

        quoted_args = re.findall(r"""(['"])(.*?)\1""", args_str)
        if quoted_args:
            args = [val for _, val in quoted_args]
            return {'tool': tool_name, 'args': args}

        log.warning(
            f"Could not parse arguments for: {tool_name}({args_str[:100]})"
        )
        return None

    def _fallback_parse_actions(self, text: str) -> List[Dict]:
        """Scan raw text for known tool call patterns when no <action> tags found."""
        actions = []
        pattern = r'\b(' + '|'.join(self.KNOWN_TOOLS) + r')\s*\((.*?)\)'
        for match in re.finditer(pattern, text, re.DOTALL):
            parsed = self._parse_single_action(match.group(0))
            if parsed:
                actions.append(parsed)
        return actions

    # ── Tool Execution ──

    def _execute_actions(self, actions: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        results = []
        if not actions:
            return results

        console.print("\n[bold green]Executing Actions...[/bold green]")
        for action in actions:
            tool_name = action.get("tool")
            tool_args = action.get("args", [])

            if tool_name in self.KNOWN_TOOLS and hasattr(self, tool_name):
                method = getattr(self, tool_name)
                try:
                    result = method(*tool_args)
                except Exception as e:
                    result = {"success": False, "error": str(e)}

                # Show truncated preview
                preview = result.get('content', result.get('error', ''))
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                console.print(f"  [cyan]{tool_name}[/cyan]: {preview}")
                results.append({
                    "tool": tool_name, "args": tool_args, "result": result,
                })

                if not result.get('success', False):
                    log.warning(
                        f"  [bold yellow]Tool '{tool_name}' failed.[/bold yellow]"
                    )
                    break
            else:
                error_result = {
                    "success": False, "error": f"Unknown tool: {tool_name}",
                }
                log.error(f"  [bold red]Unknown tool: {tool_name}[/bold red]")
                results.append({
                    "tool": tool_name, "args": tool_args, "result": error_result,
                })
                break

        return results

    # ── Query & Chat ──

    def query(self, user_input: str):
        """Process a single query against the knowledge graph."""
        console.print(f"\n[bold green]Query:[/bold green] {user_input}")

        # Assemble knowledge context via Anchor & Expand
        context = self.context_manager.assemble_context(user_input)

        if context:
            final_prompt = (
                "## Knowledge Context\n\n"
                f"{context}\n\n"
                "## User's Question\n\n"
                f"{user_input}\n\n"
                "Respond using the <think> and <action> format."
            )
        else:
            final_prompt = (
                "The knowledge graph is empty or has no relevant entities "
                "for this query.\n\n"
                "## User's Question\n\n"
                f"{user_input}\n\n"
                "Respond using the <think> and <action> format."
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
        ]

        console.print("[yellow]Thinking...[/yellow]")
        llm_response = self.llm.chat(messages)

        thoughts, actions, plain_text = self._parse_response(llm_response)

        if thoughts:
            console.print("\n[bold cyan]Reasoning:[/bold cyan]")
            console.print(Markdown(thoughts))

        if plain_text:
            console.print("\n[bold cyan]Answer:[/bold cyan]")
            console.print(Markdown(plain_text))

        if not thoughts and not plain_text:
            console.print("\n[bold cyan]Answer:[/bold cyan]")
            console.print(llm_response)

        if actions:
            self._execute_actions(actions)

    def chat(self, user_input: str, max_tool_rounds: int = 5):
        """Multi-turn agentic chat with tool use loop."""
        # Assemble context for this turn
        context = self.context_manager.assemble_context(user_input)

        if context:
            user_content = (
                "## Knowledge Context\n\n"
                f"{context}\n\n"
                "## User's Question\n\n"
                f"{user_input}"
            )
        else:
            user_content = user_input

        self.conversation_history.append({
            "role": "user", "content": user_content,
        })

        for round_num in range(max_tool_rounds):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self._trim_history())

            console.print("[yellow]Thinking...[/yellow]")
            llm_response = self.llm.chat(messages)

            thoughts, actions, plain_text = self._parse_response(llm_response)

            if thoughts:
                console.print("\n[bold cyan]Reasoning:[/bold cyan]")
                console.print(Markdown(thoughts))

            # No actions → final answer
            if not actions:
                self.conversation_history.append({
                    "role": "assistant", "content": llm_response,
                })

                if plain_text:
                    console.print("\n[bold cyan]Answer:[/bold cyan]")
                    console.print(Markdown(plain_text))
                elif not thoughts:
                    console.print("\n[bold cyan]Answer:[/bold cyan]")
                    console.print(llm_response)
                return

            # Execute tools and feed results back
            tool_results = self._execute_actions(actions)
            self.conversation_history.append({
                "role": "assistant", "content": llm_response,
            })
            tool_result_text = self._format_tool_results(tool_results)
            self.conversation_history.append({
                "role": "user", "content": tool_result_text,
            })

        console.print(
            f"\n[yellow]Reached maximum tool rounds ({max_tool_rounds}).[/yellow]"
        )

    def _trim_history(self) -> List[Dict[str, str]]:
        """Trim conversation history to stay within token budget."""
        trimmed = []
        token_count = 0

        for msg in reversed(self.conversation_history):
            msg_tokens = len(msg["content"]) // 4  # rough estimate
            if token_count + msg_tokens > self.max_history_tokens and trimmed:
                break
            trimmed.append(msg)
            token_count += msg_tokens

        trimmed.reverse()
        return trimmed

    def _format_tool_results(self, results: List[Dict]) -> str:
        """Format tool results as a message for the LLM."""
        parts = ["## Tool Results\n"]
        for r in results:
            tool = r["tool"]
            args = r["args"]
            result = r["result"]
            success = result.get("success", False)
            status = "SUCCESS" if success else "FAILED"

            parts.append(
                f"### {tool}({', '.join(repr(a) for a in args)}) — {status}"
            )

            if not success:
                parts.append(f"Error: {result.get('error', 'Unknown error')}")
            elif "content" in result:
                content = result["content"]
                if len(content) > 5000:
                    content = content[:5000] + "\n\n[... truncated ...]"
                parts.append(content)
            else:
                parts.append(str(result))

            parts.append("")

        return "\n".join(parts)

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()
