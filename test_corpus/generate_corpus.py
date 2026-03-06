#!/usr/bin/env python3
"""
Deterministic corpus generator for Cortex evaluation framework.

Reads universe.json and produces meeting transcripts, associate profiles,
project summaries, ADRs, one-on-ones, sprint planning docs, and filler
documents with corresponding .expected.json ground truth files.

Usage:
    python test_corpus/generate_corpus.py [--seed 42]

Existing hand-written documents are never overwritten.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

CORPUS_DIR = Path(__file__).parent
UNIVERSE_PATH = CORPUS_DIR / "universe.json"

# Files that already exist and must not be overwritten
EXISTING_FILES = {
    "meetings/engineering-retro.md",
    "meetings/q3-design-review.md",
    "meetings/weekly-standup-2024-11-18.md",
    "associates/john-doe.md",
    "work/dashboard-redesign.md",
}

MONTH_NAMES = {
    "2024-06": "June 2024", "2024-07": "July 2024", "2024-08": "August 2024",
    "2024-09": "September 2024", "2024-10": "October 2024", "2024-11": "November 2024",
    "2024-12": "December 2024",
    "2025-01": "January 2025", "2025-02": "February 2025", "2025-03": "March 2025",
    "2025-04": "April 2025", "2025-05": "May 2025",
}


def slugify(name: str) -> str:
    """Convert a name to a filename slug."""
    return name.lower().replace(" ", "-").replace("/", "-").replace("'", "")


def month_to_timestamp(month_str: str, day: int = 15) -> int:
    """Convert '2024-08' + day to Unix timestamp."""
    dt = datetime(int(month_str[:4]), int(month_str[5:7]), day, 10, 0, 0)
    return int(dt.timestamp())


def an_or_a(word: str) -> str:
    """Return 'an' or 'a' depending on the word."""
    return "an" if word[0].lower() in "aeiou" else "a"


# ---------------------------------------------------------------------------
# TextBuilder — clean document assembly
# ---------------------------------------------------------------------------

class TextBuilder:
    """Builds markdown documents with clean formatting."""

    def __init__(self):
        self._lines: List[str] = []

    def heading(self, text: str, level: int = 1) -> "TextBuilder":
        self._lines.append(f"{'#' * level} {text}")
        self._lines.append("")
        return self

    def paragraph(self, text: str) -> "TextBuilder":
        self._lines.append(text)
        self._lines.append("")
        return self

    def speaker(self, name: str, text: str) -> "TextBuilder":
        self._lines.append(f"**{name}:** {text}")
        self._lines.append("")
        return self

    def bullet(self, text: str) -> "TextBuilder":
        self._lines.append(f"- {text}")
        return self

    def numbered(self, n: int, text: str) -> "TextBuilder":
        self._lines.append(f"{n}. {text}")
        return self

    def blank(self) -> "TextBuilder":
        self._lines.append("")
        return self

    def build(self) -> str:
        """Build the final string, collapsing consecutive blank lines."""
        result = []
        prev_blank = False
        for line in self._lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank
        # Strip trailing blank lines
        while result and result[-1].strip() == "":
            result.pop()
        return "\n".join(result) + "\n"


# ---------------------------------------------------------------------------
# EventRegistry — cross-document reference engine
# ---------------------------------------------------------------------------

class EventRegistry:
    """Built from the full timeline; provides lookups for cross-references."""

    def __init__(self, universe: Dict):
        self.universe = universe
        self._decisions: List[Dict] = universe.get("decisions", [])
        self._incidents: List[Dict] = universe.get("incidents", [])
        self._adrs: List[Dict] = universe.get("adrs", [])
        self._projects_by_name = {p["name"]: p for p in universe["projects"]}
        self._people_by_name = {p["name"]: p for p in universe["people"]}

        # Build project membership index: person -> list of (project, role)
        self._person_projects: Dict[str, List[Tuple[str, str]]] = {}
        for proj in universe["projects"]:
            for member in [proj["lead"]] + proj["team"]:
                role = "lead" if member == proj["lead"] else "contributor"
                self._person_projects.setdefault(member, []).append((proj["name"], role))

    def decisions_before(self, month: str) -> List[Dict]:
        return [d for d in self._decisions if d["month"] < month]

    def decisions_in(self, month: str) -> List[Dict]:
        return [d for d in self._decisions if d["month"] == month]

    def active_projects_for(self, person: str, month: str) -> List[Dict]:
        """Projects a person is on that are active during the given month."""
        results = []
        for proj_name, _role in self._person_projects.get(person, []):
            proj = self._projects_by_name.get(proj_name)
            if not proj:
                continue
            if proj["start_month"] <= month:
                if proj.get("end_month") is None or proj["end_month"] >= month:
                    results.append(proj)
        return results

    def completed_projects_before(self, month: str) -> List[Dict]:
        return [p for p in self.universe["projects"]
                if p["status"] == "completed" and p.get("end_month") and p["end_month"] < month]

    def last_incident_before(self, month: str) -> Optional[Dict]:
        prior = [i for i in self._incidents if i["month"] < month]
        return prior[-1] if prior else None

    def incidents_in(self, month: str) -> List[Dict]:
        return [i for i in self._incidents if i["month"] == month]

    def get_decision(self, ref: str) -> Optional[Dict]:
        for d in self._decisions:
            if d["id"] == ref:
                return d
        return None

    def get_incident(self, ref: str) -> Optional[Dict]:
        for i in self._incidents:
            if i["id"] == ref:
                return i
        return None

    def get_adr(self, ref: str) -> Optional[Dict]:
        for a in self._adrs:
            if a["id"] == ref:
                return a
        return None

    def get_project(self, name: str) -> Optional[Dict]:
        return self._projects_by_name.get(name)

    def get_person(self, name: str) -> Optional[Dict]:
        return self._people_by_name.get(name)

    def cross_ref_for_month(self, month: str, rng: random.Random) -> Optional[str]:
        """Generate a cross-document reference sentence for this month."""
        prior_decisions = self.decisions_before(month)
        prior_incidents = [i for i in self._incidents if i["month"] < month]
        completed = self.completed_projects_before(month)

        options = []
        if prior_decisions:
            d = rng.choice(prior_decisions)
            options.append(f"This connects to our earlier decision to {d['detail'].lower()} back in {MONTH_NAMES.get(d['month'], d['month'])}.")
        if prior_incidents:
            inc = rng.choice(prior_incidents)
            options.append(f"We should keep in mind the {inc['detail'].split('—')[0].strip().lower()} incident from {MONTH_NAMES.get(inc['month'], inc['month'])}.")
        if completed:
            proj = rng.choice(completed)
            options.append(f"We can build on the lessons from the {proj['name']} project which shipped in {MONTH_NAMES.get(proj.get('end_month', ''), '')}.")

        return rng.choice(options) if options else None


# ---------------------------------------------------------------------------
# TechTaxonomy — topic-aware technology filtering
# ---------------------------------------------------------------------------

class TechTaxonomy:
    """Filters technologies by topic relevance and participant skills."""

    def __init__(self, universe: Dict):
        self._categories = universe.get("tech_categories", {})
        # Store topic mapping with lowercase keys for case-insensitive lookup
        self._topic_mapping = {k.lower(): v for k, v in universe.get("topic_tech_categories", {}).items()}
        self._all_techs = set(universe.get("technologies", []))

    def relevant_techs_for_topic(self, topic: str, participant_skills: List[str]) -> List[str]:
        """Technologies relevant to both the topic AND participants' skills."""
        # Find which tech categories apply to this topic (case-insensitive)
        cat_names = self._topic_mapping.get(topic.lower(), [])
        topic_techs = set()
        for cat in cat_names:
            topic_techs.update(self._categories.get(cat, []))

        # If no topic mapping, fall back to all technologies
        if not topic_techs:
            topic_techs = self._all_techs

        # Intersect with participant skills
        skill_set = set(participant_skills)
        relevant = list(topic_techs & skill_set)

        # If intersection is empty, fall back to topic techs only
        if not relevant:
            relevant = list(topic_techs & self._all_techs)

        return relevant if relevant else list(skill_set & self._all_techs)


# ---------------------------------------------------------------------------
# DialogueEngine — multi-turn speaker interaction
# ---------------------------------------------------------------------------

class DialogueEngine:
    """Generates dialogue where speakers respond to each other."""

    # Templates keyed by personality
    PERSONALITY_TEMPLATES = {
        "methodical": [
            "Let me walk through the data on this. {content}",
            "I've been analyzing the metrics and {content}",
            "Based on my review of the documentation, {content}",
            "I want to make sure we're systematic about this. {content}",
        ],
        "skeptic": [
            "I have concerns about this approach. {content}",
            "Before we commit, we should consider the risks. {content}",
            "I'm not fully convinced yet. {content}",
            "What's our fallback if this doesn't work? {content}",
        ],
        "enthusiast": [
            "I'm really excited about this direction. {content}",
            "This is going to be a big win for us. {content}",
            "I've been experimenting with this and the results are promising. {content}",
            "I think this could transform our workflow. {content}",
        ],
        "pragmatist": [
            "What's the simplest path to getting this done? {content}",
            "Let's focus on what we can ship this quarter. {content}",
            "I think we need to be practical here. {content}",
            "The trade-off we need to evaluate is straightforward. {content}",
        ],
        "concise": [
            "{content}",
            "Quick update: {content}",
            "Two things. {content}",
            "{content} That's it from me.",
        ],
    }

    REFERENCE_TEMPLATES = [
        "Building on what {prev} said, ",
        "To add to {prev}'s point, ",
        "I agree with {prev}. ",
        "{prev} raises a good point. ",
    ]

    TECH_DISCUSSION_TEMPLATES = [
        "I've been working with {tech} and the performance characteristics are solid — {detail}.",
        "Our {tech} setup needs attention. {detail}.",
        "The {tech} integration is coming along well. {detail}.",
        "I ran benchmarks on {tech} last week. {detail}.",
        "We should discuss our {tech} configuration. {detail}.",
    ]

    TECH_DETAILS = {
        "React": "the component re-render performance is within our 16ms budget",
        "TypeScript": "strict mode caught 15 type errors during the migration",
        "GraphQL": "query complexity scoring is preventing the N+1 issues we saw before",
        "Kafka": "the consumer lag is under 100ms even at peak throughput",
        "PostgreSQL": "the query planner is choosing index scans correctly after the ANALYZE",
        "Redis": "we're at 60% memory utilization with room to grow",
        "Kubernetes": "the HPA scaling is keeping pods between 2 and 8 replicas as expected",
        "Docker": "the multi-stage build reduced image size from 1.2GB to 340MB",
        "Datadog": "the custom dashboards are giving us real-time visibility into the pipeline",
        "Prometheus": "the scrape interval is at 15s which gives us good resolution without overhead",
        "Airflow": "the DAG execution times are consistent at about 12 minutes per run",
        "Snowflake": "the warehouse auto-scaling is keeping costs predictable",
        "PyTorch": "the model training is converging in about 45 epochs on the current dataset",
        "MLflow": "experiment tracking is working well — we have 200+ logged runs",
        "Istio": "the sidecar injection is working but adding about 3ms latency per hop",
        "Go": "the auth service is handling 10K req/s with 8MB memory footprint",
        "FastAPI": "the serving endpoint p99 is at 12ms which is well within our 50ms budget",
        "Django": "the ORM query optimization brought the endpoint from 800ms to 120ms",
        "Stripe": "the webhook retry logic is handling all edge cases correctly",
        "Firebase": "push notification delivery rate is at 98.5% across iOS and Android",
        "Next.js": "the SSR pages are loading in under 1.5 seconds on mobile",
        "Tailwind CSS": "the design tokens are consistent with the Figma specs",
        "Apollo Server": "the federation gateway is handling 5K queries/sec with caching",
        "PagerDuty": "the escalation policies are working and mean time to acknowledge is under 3 minutes",
        "Vault": "the dynamic secret rotation is running every 24 hours without issues",
        "OAuth": "token refresh flow is working correctly with the 15-minute expiry",
        "JWT": "the RSA-256 token validation adds less than 1ms overhead",
        "Grafana": "the new dashboards are being used daily by the on-call team",
        "OpenTelemetry": "the trace propagation is working across all instrumented services",
        "Jaeger": "we can trace requests end-to-end across 12 services now",
        "Envoy": "the proxy is handling L7 routing correctly with our custom filters",
        "React Native": "the shared component library is working across iOS and Android",
        "dbt": "the test coverage for data models is at 85% with schema tests on all tables",
        "RabbitMQ": "the dead letter queue is catching malformed messages as expected",
        "Spark": "the historical backfill job processed 2TB in about 4 hours",
        "Swagger": "the auto-generated docs are staying in sync with the API changes",
        "Helm": "the chart templating is correctly parameterizing per-environment configs",
        "ArgoCD": "the GitOps sync is completing within 90 seconds of merge",
        "Terraform": "the state management with S3 backend is working reliably",
        "GitHub Actions": "the CI pipeline runs in about 8 minutes end to end",
    }

    def __init__(self, registry: EventRegistry, taxonomy: TechTaxonomy, rng: random.Random):
        self.registry = registry
        self.taxonomy = taxonomy
        self.rng = rng

    def generate_discussion(
        self,
        topic: str,
        participants: List[str],
        relevant_techs: List[str],
        month: str,
        num_turns: int = 0,
    ) -> Tuple[List[str], Set[str]]:
        """Generate multi-turn discussion. Returns (speaker_lines, mentioned_techs)."""
        if num_turns == 0:
            num_turns = max(len(participants), 3)

        lines: List[str] = []
        mentioned = set()
        prev_speaker = None

        # Determine which speaker gets the cross-reference
        xref_speaker_idx = self.rng.randint(0, len(participants) - 1)

        for turn in range(num_turns):
            speaker_idx = turn % len(participants)
            speaker = participants[speaker_idx]
            person = self.registry.get_person(speaker)
            personality = person.get("personality", "pragmatist") if person else "pragmatist"

            # Pick a technology for this turn
            if relevant_techs:
                tech = relevant_techs[turn % len(relevant_techs)]
            else:
                tech = "the system"

            detail = self.TECH_DETAILS.get(tech, "we're making good progress on the implementation")
            # Capitalize detail since it follows a sentence boundary (period or dash)
            detail = detail[0].upper() + detail[1:] if detail else detail
            content_template = self.rng.choice(self.TECH_DISCUSSION_TEMPLATES)
            content = content_template.format(tech=tech, detail=detail)
            mentioned.add(tech)

            # Build the line with personality wrapping
            templates = self.PERSONALITY_TEMPLATES.get(personality, self.PERSONALITY_TEMPLATES["pragmatist"])
            wrapper = self.rng.choice(templates)
            # If content is inserted mid-sentence (after "and ", ", "), lowercase first char
            insert_pos = wrapper.find("{content}")
            if insert_pos > 0:
                before = wrapper[:insert_pos]
                if before.rstrip().endswith(("and", ",")):
                    content = content[0].lower() + content[1:]
            line = wrapper.format(content=content)

            # Maybe reference previous speaker
            if prev_speaker and self.rng.random() < 0.5:
                ref = self.rng.choice(self.REFERENCE_TEMPLATES).format(prev=prev_speaker)
                line = ref + line[0].lower() + line[1:]

            lines.append(f"**{speaker}:** {line}")

            # Cross-document reference for one speaker
            if turn == xref_speaker_idx:
                xref = self.registry.cross_ref_for_month(month, self.rng)
                if xref:
                    lines.append(f"**{speaker}:** {xref}")

            prev_speaker = speaker

        return lines, mentioned


# ---------------------------------------------------------------------------
# CorpusGenerator — main orchestrator
# ---------------------------------------------------------------------------

class CorpusGenerator:
    def __init__(self, seed: int = 42):
        with open(UNIVERSE_PATH) as f:
            self.universe = json.load(f)
        self.rng = random.Random(seed)
        self.people_by_name = {p["name"]: p for p in self.universe["people"]}
        self.projects_by_name = {p["name"]: p for p in self.universe["projects"]}
        self.registry = EventRegistry(self.universe)
        self.taxonomy = TechTaxonomy(self.universe)
        self.dialogue = DialogueEngine(self.registry, self.taxonomy, self.rng)

    def generate_all(self):
        """Generate the full corpus."""
        meetings = self._generate_meetings()
        associates = self._generate_associates()
        projects = self._generate_projects()
        adrs = self._generate_adrs()
        cancelled = self._generate_cancelled_projects()

        total_gen = len(meetings) + len(associates) + len(projects) + len(adrs) + len(cancelled)
        print(f"Generated: {len(meetings)} meetings, {len(associates)} associates, "
              f"{len(projects)} project summaries, {len(adrs)} ADRs, {len(cancelled)} cancelled")
        print(f"Total: {total_gen} documents (+ {len(EXISTING_FILES)} existing = {total_gen + len(EXISTING_FILES)} total)")

        self._generate_queries()

    def _should_skip(self, relative_path: str) -> bool:
        return relative_path in EXISTING_FILES

    def _write_doc(self, relative_path: str, content: str, expected: Dict):
        doc_path = CORPUS_DIR / relative_path
        expected_path = doc_path.with_suffix(".expected.json")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(content)
        expected_path.write_text(json.dumps(expected, indent=4) + "\n")

    # -------------------------------------------------------------------------
    # Helper: collect all participant skills as tech list
    # -------------------------------------------------------------------------

    def _participant_skills(self, names: List[str]) -> List[str]:
        skills = []
        for name in names:
            person = self.people_by_name.get(name, {})
            skills.extend(person.get("skills", []))
        return skills

    def _tech_skills_for(self, name: str) -> List[str]:
        person = self.people_by_name.get(name, {})
        return [s for s in person.get("skills", []) if s in self.universe["technologies"]]

    def _dedupe_entities(self, entities: List[Dict]) -> List[Dict]:
        seen: Set[Tuple[str, str]] = set()
        result = []
        for e in entities:
            key = (e["name"], e["type"])
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    # -------------------------------------------------------------------------
    # Meeting generation (topic meetings, kickoffs, postmortems, standups,
    # one-on-ones, sprint planning, filler)
    # -------------------------------------------------------------------------

    def _generate_meetings(self) -> List[str]:
        generated = []

        for month_entry in self.universe["timeline"]:
            month = month_entry["month"]
            events = month_entry["events"]

            for event in events:
                etype = event["type"]

                if etype == "meeting":
                    result = self._generate_single_meeting(event, month)
                    if result:
                        generated.append(result)

                elif etype == "project_kickoff":
                    result = self._generate_kickoff_meeting(event, month)
                    if result:
                        generated.append(result)

                elif etype == "incident":
                    ref = event.get("ref")
                    if ref:
                        incident = self.registry.get_incident(ref)
                        if incident:
                            result = self._generate_incident_postmortem(incident, month)
                            if result:
                                generated.append(result)

                elif etype == "one_on_one":
                    result = self._generate_one_on_one(event, month)
                    if result:
                        generated.append(result)

                elif etype == "sprint_planning":
                    result = self._generate_sprint_planning(event, month)
                    if result:
                        generated.append(result)

                elif etype == "filler":
                    result = self._generate_filler(event, month)
                    if result:
                        generated.append(result)

                elif etype == "standup_weeks":
                    weeks = event.get("standup_weeks", 4)
                    facilitator = event.get("facilitator", "Sarah Chen")
                    for week in range(1, weeks + 1):
                        result = self._generate_standup(month, week, facilitator)
                        if result:
                            generated.append(result)

        return generated

    def _generate_single_meeting(self, event: Dict, month: str) -> Optional[str]:
        """Generate a topic-focused meeting transcript."""
        topic = event["topic"]
        participants = event["participants"]
        slug = slugify(topic)
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        entities: List[Dict] = []
        cooccurrences: List[Dict] = []

        for name in participants:
            entities.append({"name": name, "type": "person"})

        # Use taxonomy-filtered tech
        all_skills = self._participant_skills(participants)
        relevant_techs = self.taxonomy.relevant_techs_for_topic(topic, all_skills)
        if not relevant_techs:
            relevant_techs = [s for s in all_skills if s in self.universe["technologies"]]

        # Generate dialogue
        discussion_lines, mentioned_tech = self.dialogue.generate_discussion(
            topic, participants, relevant_techs, month
        )

        for tech in mentioned_tech:
            entities.append({"name": tech, "type": "technology"})
            # Co-occur tech with speakers who have it as a skill
            for name in participants:
                if tech in self._tech_skills_for(name):
                    cooccurrences.append({"entity_a": name, "entity_b": tech})

        # Find related projects
        mentioned_projects = set()
        for name in participants:
            active = self.registry.active_projects_for(name, month)
            for proj in active:
                if self.rng.random() < 0.4:
                    mentioned_projects.add(proj["name"])
                    cooccurrences.append({"entity_a": name, "entity_b": proj["name"]})

        for proj_name in mentioned_projects:
            entities.append({"name": proj_name, "type": "project"})

        # Build document
        tb = TextBuilder()
        tb.heading(f"{topic.title()} — {date_str}")
        tb.heading("Attendees", 2)
        tb.paragraph(", ".join(participants))
        tb.heading("Discussion", 2)
        for line in discussion_lines:
            tb.paragraph(line)

        # Decisions section if there are decisions this month related to participants
        month_decisions = self.registry.decisions_in(month)
        relevant_decisions = [d for d in month_decisions
                             if any(p in d.get("participants", []) for p in participants)]
        if relevant_decisions:
            tb.heading("Decisions", 2)
            for d in relevant_decisions[:2]:
                decider = d["participants"][0] if d["participants"] else participants[0]
                tb.speaker(decider, f"Based on this discussion, we've decided: {d['detail']}. {d.get('rationale', '')}")

        # Project references
        if mentioned_projects:
            tb.heading("Related Projects", 2)
            for proj_name in list(mentioned_projects)[:3]:
                proj = self.projects_by_name.get(proj_name, {})
                tb.paragraph(f"The {proj_name} project ({proj.get('status', 'active')}) is relevant to this discussion.")

        # Action items
        tb.heading("Action Items", 2)
        action_techs = list(mentioned_tech)
        for i, p in enumerate(participants[:3]):
            tech = action_techs[i % len(action_techs)] if action_techs else "the system"
            tb.bullet(f"{p} will investigate {tech} configuration and report findings by end of week")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": max(3, len(participants)),
            "expected_chunks_max": max(8, len(participants) * 3),
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_kickoff_meeting(self, event: Dict, month: str) -> Optional[str]:
        """Generate a project kickoff meeting with tech rationale."""
        project_name = event["project"]
        participants = event["participants"]
        slug = f"{slugify(project_name)}-kickoff"
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        project = self.projects_by_name.get(project_name, {})
        proj_tech = project.get("technologies", [])
        tech_rationale = project.get("tech_rationale", {})
        goal = project.get("goal", f"deliver the {project_name} initiative")

        entities: List[Dict] = [{"name": project_name, "type": "project"}]
        cooccurrences: List[Dict] = []

        for name in participants:
            entities.append({"name": name, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": project_name})

        for tech in proj_tech[:5]:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": project_name, "entity_b": tech})

        # Build document
        tb = TextBuilder()
        tb.heading(f"{project_name} — Kickoff Meeting — {date_str}")
        tb.heading("Attendees", 2)
        tb.paragraph(", ".join(participants))

        tb.heading("Project Overview", 2)
        if participants:
            tb.speaker(participants[0],
                       f"Welcome everyone. Today we're kicking off the {project_name} project. "
                       f"Our goal is to {goal}. Let me walk through the approach and timeline.")

        tb.heading("Technical Approach", 2)
        for i, tech in enumerate(proj_tech[:5]):
            speaker = participants[i % len(participants)]
            rationale = tech_rationale.get(tech, f"proven track record and team familiarity")
            # Pick the person who has this tech as a skill to present it
            tech_owner = speaker
            for p in participants:
                if tech in self._tech_skills_for(p):
                    tech_owner = p
                    break
            tb.speaker(tech_owner,
                       f"We'll be using {tech} for this project. The rationale is {rationale}.")
            cooccurrences.append({"entity_a": tech_owner, "entity_b": tech})

        # Cross-reference prior work
        xref = self.registry.cross_ref_for_month(month, self.rng)
        if xref and participants:
            tb.speaker(participants[0], xref)

        tb.heading("Timeline", 2)
        tb.bullet(f"Phase 1: Architecture and design ({MONTH_NAMES.get(month, month)})")
        tb.bullet("Phase 2: Core implementation (following month)")
        tb.bullet("Phase 3: Testing and rollout (month after)")
        tb.blank()

        tb.heading("Action Items", 2)
        for p in participants[:3]:
            person = self.people_by_name.get(p, {})
            focus = self.rng.choice(proj_tech[:3]) if proj_tech else "architecture"
            tb.bullet(f"{p} will draft the {focus} section of the technical design doc by Friday")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 4,
            "expected_chunks_max": 12,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_incident_postmortem(self, incident: Dict, month: str) -> Optional[str]:
        """Generate an incident postmortem from structured incident data."""
        detail = incident["detail"]
        participants = incident["participants"]
        slug = slugify(detail.split("—")[0].strip()) if "—" in detail else slugify(detail[:40])
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        relative_path = f"meetings/{slug}-postmortem.md"

        if self._should_skip(relative_path):
            return None

        root_cause = incident.get("root_cause", "Under investigation")
        affected_tech = incident.get("affected_tech", [])
        affected_service = incident.get("affected_service", "Unknown service")
        resolution = incident.get("resolution", "Issue was resolved by the on-call team")

        entities: List[Dict] = []
        cooccurrences: List[Dict] = []

        for name in participants:
            entities.append({"name": name, "type": "person"})

        for tech in affected_tech:
            entities.append({"name": tech, "type": "technology"})
            for name in participants:
                if tech in self._tech_skills_for(name):
                    cooccurrences.append({"entity_a": name, "entity_b": tech})

        if affected_service:
            entities.append({"name": affected_service, "type": "project"})
            for name in participants:
                cooccurrences.append({"entity_a": name, "entity_b": affected_service})

        # Cross-reference prior incidents
        prior_incident = self.registry.last_incident_before(month)

        tb = TextBuilder()
        tb.heading(f"Incident Postmortem — {detail} — {date_str}")
        tb.heading("Attendees", 2)
        tb.paragraph(", ".join(participants))

        tb.heading("Incident Summary", 2)
        tb.paragraph(f"{detail}. This affected the {affected_service} service. "
                     f"The incident was detected by automated monitoring and the team responded within 5 minutes.")

        tb.heading("Root Cause Analysis", 2)
        if participants:
            tb.speaker(participants[0],
                       f"The root cause was identified: {root_cause}")
        if len(affected_tech) > 1:
            tech_str = ", ".join(affected_tech)
            tb.paragraph(f"The affected components included {tech_str}.")

        tb.heading("Resolution", 2)
        if len(participants) > 1:
            tb.speaker(participants[1], resolution)

        # Cross-reference prior incident
        if prior_incident and len(participants) > 1:
            tb.heading("Lessons from Prior Incidents", 2)
            tb.speaker(participants[-1],
                       f"This is similar to the {prior_incident['detail'].split('—')[0].strip().lower()} "
                       f"incident from {MONTH_NAMES.get(prior_incident['month'], prior_incident['month'])}. "
                       f"We should review whether the mitigations from that incident apply here.")

        tb.heading("Action Items", 2)
        for i, p in enumerate(participants[:3]):
            tech = affected_tech[i % len(affected_tech)] if affected_tech else "the affected service"
            tb.bullet(f"{p} will implement safeguards for {tech} to prevent recurrence")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 10,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_standup(self, month: str, week: int, facilitator: str) -> Optional[str]:
        """Generate a weekly standup meeting."""
        day = min(week * 7, 28)
        date_str = f"{month}-{day:02d}"
        slug = f"standup-{date_str}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        # Pick 3-5 engineers for this standup
        all_engineers = [p for p in self.universe["people"]
                        if p["role"] not in ("Product Manager", "UX Designer",
                                             "Technical Writer", "Engineering Director")]
        participants = self.rng.sample(all_engineers, min(len(all_engineers), self.rng.randint(3, 5)))

        entities: List[Dict] = []
        cooccurrences: List[Dict] = []

        for p in participants:
            entities.append({"name": p["name"], "type": "person"})

        tb = TextBuilder()
        tb.heading(f"Weekly Standup — {date_str}")

        for p in participants:
            # Look up their active projects
            active = self.registry.active_projects_for(p["name"], month)
            tech_skills = self._tech_skills_for(p["name"])

            if active and tech_skills:
                proj = self.rng.choice(active)
                # Pick a tech that's both in their skills AND the project's tech
                proj_techs = set(proj.get("technologies", []))
                overlap = [t for t in tech_skills if t in proj_techs]
                tech = self.rng.choice(overlap) if overlap else self.rng.choice(tech_skills)

                entities.append({"name": tech, "type": "technology"})
                entities.append({"name": proj["name"], "type": "project"})
                cooccurrences.append({"entity_a": p["name"], "entity_b": tech})
                cooccurrences.append({"entity_a": p["name"], "entity_b": proj["name"]})

                detail = self.dialogue.TECH_DETAILS.get(tech, "making steady progress")

                # ~20% chance of blocker
                if self.rng.random() < 0.2:
                    blocker = self.rng.choice([
                        f"Blocked on waiting for {tech} environment access",
                        f"Hit a compatibility issue between {tech} and our staging setup",
                        f"Need input from the {proj['name']} team on the API contract",
                    ])
                    tb.speaker(p["name"],
                               f"Working on {proj['name']}. {blocker}. "
                               f"Once unblocked, I'll continue with the {tech} integration.")
                else:
                    tb.speaker(p["name"],
                               f"Yesterday I worked on the {tech} integration for {proj['name']}. "
                               f"{detail.capitalize() if detail[0].islower() else detail}. "
                               f"Today I'll continue with testing.")
            elif tech_skills:
                tech = self.rng.choice(tech_skills)
                entities.append({"name": tech, "type": "technology"})
                cooccurrences.append({"entity_a": p["name"], "entity_b": tech})
                tb.speaker(p["name"],
                           f"I've been debugging an issue with {tech}. Found the root cause — "
                           f"it was a version mismatch. Fixed now.")
            else:
                tb.speaker(p["name"],
                           "Working on code reviews and documentation updates. No blockers.")

        # Facilitator wrap-up — reference upcoming event if possible
        upcoming_ref = ""
        month_decisions = self.registry.decisions_in(month)
        if month_decisions:
            d = self.rng.choice(month_decisions)
            upcoming_ref = f" Reminder: we have the {d['detail'].split(' ')[0].lower()} discussion coming up this week."

        tb.speaker(facilitator,
                   f"Good updates everyone. See you next week.{upcoming_ref}")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": len(participants),
            "expected_chunks_max": len(participants) * 2 + 1,
            "expected_filler_chunks_min": 1,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_one_on_one(self, event: Dict, month: str) -> Optional[str]:
        """Generate a one-on-one meeting between manager and report."""
        manager = event["manager"]
        report = event["report"]
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        slug = f"1on1-{slugify(manager)}-{slugify(report)}-{date_str}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        manager_data = self.people_by_name.get(manager, {})
        report_data = self.people_by_name.get(report, {})

        entities: List[Dict] = [
            {"name": manager, "type": "person"},
            {"name": report, "type": "person"},
        ]
        cooccurrences: List[Dict] = [{"entity_a": manager, "entity_b": report}]

        # Report's active projects
        active_projects = self.registry.active_projects_for(report, month)
        report_techs = self._tech_skills_for(report)

        tb = TextBuilder()
        tb.heading(f"1:1 — {manager} / {report} — {date_str}")

        # Check-in
        tb.heading("Check-in", 2)
        check_in_options = [
            f"How are things going this week?",
            f"What's been on your mind since we last spoke?",
            f"Any wins or frustrations to share?",
        ]
        tb.speaker(manager, self.rng.choice(check_in_options))
        report_checkin = [
            f"Things are going well overall. I've been heads-down on the implementation work.",
            f"It's been a productive week. I feel like I'm getting into a good rhythm.",
            f"Honestly, it's been a bit intense, but I'm managing.",
        ]
        tb.speaker(report, self.rng.choice(report_checkin))

        # Project status
        if active_projects:
            tb.heading("Project Updates", 2)
            proj = active_projects[0]
            entities.append({"name": proj["name"], "type": "project"})
            cooccurrences.append({"entity_a": report, "entity_b": proj["name"]})

            proj_tech = proj.get("technologies", [])
            overlap = [t for t in report_techs if t in proj_tech]
            if overlap:
                tech = self.rng.choice(overlap)
                entities.append({"name": tech, "type": "technology"})
                cooccurrences.append({"entity_a": report, "entity_b": tech})
                tb.speaker(report,
                           f"On {proj['name']}, I've been focused on the {tech} piece. "
                           f"{self.dialogue.TECH_DETAILS.get(tech, 'Making good progress')}.")
            else:
                tb.speaker(report, f"On {proj['name']}, I'm making steady progress on my tasks.")

            tb.speaker(manager, f"That's good to hear. Is there anything I can unblock for you on {proj['name']}?")
            tb.speaker(report, "I think I'm good for now. I'll reach out if I hit any blockers.")

        # Growth and feedback
        tb.heading("Growth & Development", 2)
        # Reference mentorship if applicable
        mentees = manager_data.get("mentees", [])
        if report in mentees:
            tb.speaker(manager,
                       f"I want to check in on your development. You've been growing a lot since you joined. "
                       f"What areas do you want to focus on next?")
            growth_areas = [
                f"I'd like to get more experience with system design. Maybe lead a design doc?",
                f"I want to improve my code review skills. I've been learning a lot from the team's reviews.",
                f"I'm interested in learning more about our infrastructure setup. Could I shadow someone on on-call?",
            ]
            tb.speaker(report, self.rng.choice(growth_areas))
        else:
            tb.speaker(manager, "Any feedback or areas where you'd like more support?")
            tb.speaker(report, "I appreciate the autonomy. The team has been great about answering questions when I need help.")

        # Cross-reference past events
        xref = self.registry.cross_ref_for_month(month, self.rng)
        if xref:
            tb.speaker(manager, xref)

        # Action items
        tb.heading("Action Items", 2)
        tb.bullet(f"{manager} will follow up on development opportunities discussed")
        if active_projects:
            tb.bullet(f"{report} will share a status update on {active_projects[0]['name']} by end of week")
        else:
            tb.bullet(f"{report} will share a written update on current work by end of week")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 8,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_sprint_planning(self, event: Dict, month: str) -> Optional[str]:
        """Generate a sprint planning meeting."""
        project_name = event["project"]
        sprint_num = event["sprint"]
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        slug = f"sprint-planning-{slugify(project_name)}-sprint-{sprint_num}-{date_str}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        project = self.projects_by_name.get(project_name, {})
        team = [project.get("lead", "")] + project.get("team", [])
        team = [t for t in team if t]  # filter empty
        proj_tech = project.get("technologies", [])
        goal = project.get("goal", f"deliver the {project_name}")

        entities: List[Dict] = [{"name": project_name, "type": "project"}]
        cooccurrences: List[Dict] = []

        for name in team:
            entities.append({"name": name, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": project_name})

        for tech in proj_tech[:3]:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": project_name, "entity_b": tech})

        tb = TextBuilder()
        tb.heading(f"Sprint Planning — {project_name} — Sprint {sprint_num} — {date_str}")
        tb.heading("Attendees", 2)
        tb.paragraph(", ".join(team))

        tb.heading("Sprint Goal", 2)
        sprint_goals = [
            f"Complete the {proj_tech[0] if proj_tech else 'core'} integration for {project_name}",
            f"Ship the {project_name} beta to internal users for feedback",
            f"Resolve remaining blockers and prepare {project_name} for load testing",
            f"Finalize the {proj_tech[-1] if proj_tech else 'main'} layer and begin end-to-end testing",
        ]
        selected_goal = sprint_goals[(sprint_num - 1) % len(sprint_goals)]
        if team:
            tb.speaker(team[0], f"Our goal for sprint {sprint_num}: {selected_goal}. "
                       f"This ties back to our overall objective to {goal}.")

        tb.heading("Backlog Items", 2)
        for i, tech in enumerate(proj_tech[:4]):
            assignee = team[i % len(team)]
            story_points = self.rng.choice([2, 3, 5, 8])
            backlog_items = [
                f"Implement {tech} connection handling and retry logic ({story_points} pts)",
                f"Write integration tests for {tech} module ({story_points} pts)",
                f"Update {tech} configuration for production environment ({story_points} pts)",
                f"Performance optimization for {tech} queries ({story_points} pts)",
            ]
            item = backlog_items[i % len(backlog_items)]
            tb.bullet(f"[{assignee}] {item}")
            cooccurrences.append({"entity_a": assignee, "entity_b": tech})
        tb.blank()

        tb.heading("Capacity", 2)
        for member in team:
            days = self.rng.choice([8, 9, 10])
            tb.bullet(f"{member}: {days} days available")
        tb.blank()

        # Cross-project dependencies
        completed = self.registry.completed_projects_before(month)
        if completed:
            dep = self.rng.choice(completed)
            tb.heading("Dependencies", 2)
            tb.paragraph(f"We have a dependency on the {dep['name']} project's output. "
                        f"Since that shipped in {MONTH_NAMES.get(dep.get('end_month', ''), '')}, "
                        f"we can proceed with integration.")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 10,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_filler(self, event: Dict, month: str) -> Optional[str]:
        """Generate a low-information filler meeting (lunch, office hours, etc.)."""
        slug = event["slug"]
        title = event["title"]
        participants = event["participants"]
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        relative_path = f"meetings/filler-{slug}-{date_str}.md"

        if self._should_skip(relative_path):
            return None

        entities: List[Dict] = []
        cooccurrences: List[Dict] = []

        for name in participants:
            entities.append({"name": name, "type": "person"})

        # Filler meetings have minimal entity content
        tb = TextBuilder()
        tb.heading(f"{title} — {date_str}")
        tb.heading("Attendees", 2)
        tb.paragraph(", ".join(participants))

        # Low-information content that shouldn't match entity queries
        filler_content = {
            "team-lunch": [
                "Casual team lunch. Discussed weekend plans and the new coffee machine in the break room.",
                "Good to catch up outside of work context. The restaurant was great — we should go back.",
            ],
            "holiday-party": [
                "Planning the end-of-year celebration. Discussed venue options, food preferences, and activity ideas.",
                "Budget is approved. Need to finalize the guest list and send invitations.",
            ],
            "book-club": [
                "Discussed this month's book. Good conversation about the practical applications.",
                "Next month we're reading something lighter. Nominations are open.",
            ],
            "office-hours": [
                "Open Q&A session. A few questions came up about development setup and tooling.",
                "Covered some common issues and shared some tips. Good attendance this week.",
            ],
            "hackathon": [
                "Spring hackathon kickoff. Teams formed and project ideas pitched.",
                "Lots of creative ideas this year. Presentations are Friday afternoon.",
            ],
            "offsite": [
                "Planning session for the upcoming team offsite. Discussing agenda and logistics.",
                "Finalizing the venue and travel arrangements. Team building activities TBD.",
            ],
            "tech-talk": [
                "Internal tech talk. Good presentation and discussion from the audience.",
                "Recording will be posted on the internal wiki for those who couldn't attend.",
            ],
            "new-year": [
                "New year team kickoff. Reviewed accomplishments from last year and set high-level goals.",
                "Good energy to start the year. Individual goal setting to follow in 1:1s.",
            ],
            "intern": [
                "Welcome session for the summer intern cohort. Covered company overview and logistics.",
                "Mentors assigned. First week focused on environment setup and onboarding docs.",
            ],
            "design-review": [
                "Informal design review and social. Shared recent work and got feedback over snacks.",
                "Good cross-team collaboration. A few ideas to follow up on.",
            ],
        }

        # Match slug prefix to content
        content_key = None
        for key in filler_content:
            if key in slug:
                content_key = key
                break

        paragraphs = filler_content.get(content_key, [
            "Brief team sync. No major action items.",
            "Casual discussion. Will follow up on a couple of items via Slack.",
        ])

        for para in paragraphs:
            tb.paragraph(para)

        tb.heading("Notes", 2)
        tb.paragraph("No formal action items from this meeting.")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 2,
            "expected_chunks_max": 5,
            "expected_filler_chunks": len(paragraphs),
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # Associate profile generation
    # -------------------------------------------------------------------------

    def _generate_associates(self) -> List[str]:
        generated = []
        for person in self.universe["people"]:
            slug = slugify(person["name"])
            relative_path = f"associates/{slug}.md"
            if self._should_skip(relative_path):
                continue
            result = self._generate_associate_profile(person)
            if result:
                generated.append(result)
        return generated

    def _generate_associate_profile(self, person: Dict) -> Optional[str]:
        name = person["name"]
        slug = slugify(name)
        relative_path = f"associates/{slug}.md"

        entities: List[Dict] = [{"name": name, "type": "person"}]
        cooccurrences: List[Dict] = []

        # Organization
        entities.append({"name": "TechCorp", "type": "organization"})
        cooccurrences.append({"entity_a": name, "entity_b": "TechCorp"})

        if person.get("team"):
            entities.append({"name": person["team"], "type": "organization"})
            cooccurrences.append({"entity_a": name, "entity_b": person["team"]})

        # Technologies
        tech_skills = [s for s in person["skills"] if s in self.universe["technologies"]]
        for tech in tech_skills:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": name, "entity_b": tech})

        # Projects
        person_projects = []
        for proj in self.universe["projects"]:
            if name in ([proj["lead"]] + proj["team"]):
                person_projects.append(proj)
                entities.append({"name": proj["name"], "type": "project"})
                cooccurrences.append({"entity_a": name, "entity_b": proj["name"]})

        # Collaborators
        collaborators = set()
        for proj in person_projects:
            for member in [proj["lead"]] + proj["team"]:
                if member != name:
                    collaborators.add(member)

        collab_list = list(collaborators)
        mentioned_collabs = self.rng.sample(collab_list, min(len(collab_list), 3))
        for collab in mentioned_collabs:
            entities.append({"name": collab, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": collab})

        # Build profile
        tb = TextBuilder()
        tb.heading(f"{name} — {person['role']}")

        tb.heading("Overview", 2)
        role = person["role"]
        article = an_or_a(role)
        tb.paragraph(f"{name} is {article} {role} at TechCorp who joined in "
                    f"{person['joined']}. They work on the {person['team']} team.")

        # Skills
        tb.heading("Skills & Expertise", 2)
        if tech_skills:
            tb.paragraph(f"{name} specializes in {', '.join(tech_skills[:3])}."
                        + (f" They also have experience with {', '.join(tech_skills[3:])}."
                           if len(tech_skills) > 3 else ""))

        soft_skills = person.get("soft_skills", [])
        if soft_skills:
            tb.paragraph(f"Beyond technical skills, {name} brings strengths in "
                        f"{', '.join(soft_skills)}.")

        # Projects
        if person_projects:
            tb.heading("Current Projects", 2)
            for proj in person_projects:
                role_str = "leading" if proj["lead"] == name else "contributing to"
                tech_str = f", which uses {', '.join(proj['technologies'][:3])}" if proj["technologies"] else ""
                tb.paragraph(f"{name} is currently {role_str} the {proj['name']} project{tech_str}.")

        # Notable contributions
        notable = person.get("notable_work", [])
        if notable:
            tb.heading("Notable Contributions", 2)
            for item in notable:
                tb.bullet(item)
            tb.blank()

        # Mentorship
        mentees = person.get("mentees", [])
        mentor = person.get("mentors")
        if mentees or mentor:
            tb.heading("Mentorship", 2)
            if mentees:
                tb.paragraph(f"{name} mentors {', '.join(mentees)}.")
                for mentee in mentees:
                    if mentee not in [e["name"] for e in entities]:
                        entities.append({"name": mentee, "type": "person"})
                    cooccurrences.append({"entity_a": name, "entity_b": mentee})
            if mentor:
                tb.paragraph(f"{name} is mentored by {mentor}.")
                if mentor not in [e["name"] for e in entities]:
                    entities.append({"name": mentor, "type": "person"})
                cooccurrences.append({"entity_a": name, "entity_b": mentor})

        # Collaborators
        if mentioned_collabs:
            tb.heading("Key Collaborators", 2)
            tb.paragraph(f"{name} works closely with {', '.join(mentioned_collabs)}.")

        # Working style
        tb.heading("Working Style", 2)
        personality = person.get("personality", "pragmatist")
        style_map = {
            "methodical": f"{name} is methodical and thorough, preferring detailed documentation and careful code reviews before moving forward.",
            "skeptic": f"{name} is known for asking tough questions and pushing back on assumptions, ensuring the team considers all angles.",
            "enthusiast": f"{name} brings high energy to projects and is often the first to experiment with new tools and approaches.",
            "pragmatist": f"{name} focuses on practical outcomes, preferring simple solutions that ship on time over perfect ones that don't.",
            "concise": f"{name} is direct and efficient in communication, keeping meetings short and documentation focused.",
        }
        tb.paragraph(style_map.get(personality, style_map["pragmatist"]))

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 10,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # Project summary generation
    # -------------------------------------------------------------------------

    def _generate_projects(self) -> List[str]:
        generated = []
        for project in self.universe["projects"]:
            if project.get("status") == "cancelled":
                continue  # Handled separately
            slug = slugify(project["name"])
            relative_path = f"work/{slug}.md"
            if self._should_skip(relative_path):
                continue
            result = self._generate_project_summary(project)
            if result:
                generated.append(result)
        return generated

    def _generate_project_summary(self, project: Dict) -> Optional[str]:
        name = project["name"]
        slug = slugify(name)
        relative_path = f"work/{slug}.md"

        entities: List[Dict] = [{"name": name, "type": "project"}]
        cooccurrences: List[Dict] = []

        lead = project["lead"]
        entities.append({"name": lead, "type": "person"})
        cooccurrences.append({"entity_a": name, "entity_b": lead})

        for member in project["team"]:
            entities.append({"name": member, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": member})

        for tech in project["technologies"]:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": name, "entity_b": tech})

        tech_rationale = project.get("tech_rationale", {})
        goal = project.get("goal", "")
        outcomes = project.get("outcomes")

        tb = TextBuilder()
        tb.heading(f"{name} — Project Summary")

        tb.heading("Project Overview", 2)
        team_str = ", ".join(project["team"][:3])
        start = MONTH_NAMES.get(project["start_month"], project["start_month"])
        tb.paragraph(f"The {name} project was initiated in {start} and is led by {lead}. "
                    f"The core team includes {team_str}.")
        if goal:
            tb.paragraph(f"Goal: {goal}.")

        tb.heading("Technical Architecture", 2)
        if project["technologies"]:
            tech_str = ", ".join(project["technologies"][:4])
            tb.paragraph(f"The technical stack includes {tech_str}.")
            for tech in project["technologies"][:4]:
                rationale = tech_rationale.get(tech)
                if rationale:
                    tb.paragraph(f"{tech} was selected for {rationale}.")
                else:
                    reasons = [
                        f"{tech} was chosen for its strong community support and proven track record in production.",
                        f"The team selected {tech} based on performance benchmarks and developer experience.",
                        f"{tech} integrates well with the existing infrastructure.",
                    ]
                    tb.paragraph(self.rng.choice(reasons))

        # Outcomes for completed projects
        if outcomes:
            tb.heading("Outcomes", 2)
            tb.paragraph(f"Results: {outcomes}.")

        tb.heading("Key Decisions", 2)
        # Find decisions related to this project's participants or technologies
        related_decisions = [d for d in self.universe.get("decisions", [])
                           if any(p in d.get("participants", [])
                                  for p in [lead] + project["team"])]
        if related_decisions:
            for d in related_decisions[:2]:
                tb.numbered(1, f"{d['detail']} ({MONTH_NAMES.get(d['month'], d['month'])})")
        else:
            tb.numbered(1, f"Adopted {project['technologies'][0] if project['technologies'] else 'agile'} as the primary framework")
            tb.numbered(2, "Implemented CI/CD pipeline using GitHub Actions")
        tb.blank()

        tb.heading("Timeline", 2)
        tb.bullet(f"{start}: Project kickoff and initial architecture design")
        if project["end_month"]:
            end = MONTH_NAMES.get(project["end_month"], project["end_month"])
            tb.bullet(f"{end}: Project completion and production rollout")
        else:
            tb.bullet("Ongoing: Active development and iteration")
        tb.blank()

        tb.heading("Status", 2)
        status_map = {
            "completed": "The project has been successfully completed and is in production.",
            "in_progress": "The project is actively under development.",
            "planning": "The project is in the planning phase with design work underway.",
        }
        tb.paragraph(status_map.get(project["status"], "Status unknown."))

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 4,
            "expected_chunks_max": 12,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # ADR generation
    # -------------------------------------------------------------------------

    def _generate_adrs(self) -> List[str]:
        generated = []
        for adr in self.universe.get("adrs", []):
            result = self._generate_adr(adr)
            if result:
                generated.append(result)
        return generated

    def _generate_adr(self, adr: Dict) -> Optional[str]:
        number = adr["number"]
        title = adr["title"]
        slug = f"adr-{number:03d}-{slugify(title)}"
        relative_path = f"work/{slug}.md"

        if self._should_skip(relative_path):
            return None

        author = adr["author"]
        prompted_by = adr.get("prompted_by")
        context = adr.get("context", "")
        decision_text = adr.get("decision", "")
        rationale = adr.get("rationale", "")
        alternatives = adr.get("alternatives_considered", [])
        consequences = adr.get("consequences", [])
        status = adr.get("status", "accepted")
        month = adr["month"]

        entities: List[Dict] = [{"name": author, "type": "person"}]
        cooccurrences: List[Dict] = []

        # Extract tech from title and decision
        for tech in self.universe["technologies"]:
            if tech in title or tech in decision_text or tech in context:
                entities.append({"name": tech, "type": "technology"})
                cooccurrences.append({"entity_a": author, "entity_b": tech})

        # Cross-reference to prompting event
        cross_ref = ""
        if prompted_by:
            if prompted_by.startswith("incident_"):
                incident = self.registry.get_incident(prompted_by)
                if incident:
                    cross_ref = f"This ADR was prompted by the {incident['detail']} in {MONTH_NAMES.get(incident['month'], incident['month'])}."
            elif prompted_by.startswith("decision_"):
                decision = self.registry.get_decision(prompted_by)
                if decision:
                    cross_ref = f"This ADR formalizes the decision to {decision['detail'].lower()} made in {MONTH_NAMES.get(decision['month'], decision['month'])}."

        tb = TextBuilder()
        tb.heading(f"ADR-{number:03d}: {title}")
        tb.paragraph(f"**Status:** {status.title()}")
        tb.paragraph(f"**Date:** {MONTH_NAMES.get(month, month)}")
        tb.paragraph(f"**Author:** {author}")

        tb.heading("Context", 2)
        tb.paragraph(context)
        if cross_ref:
            tb.paragraph(cross_ref)

        tb.heading("Decision", 2)
        tb.paragraph(decision_text)
        if rationale:
            tb.paragraph(f"**Rationale:** {rationale}")

        if alternatives:
            tb.heading("Alternatives Considered", 2)
            for alt in alternatives:
                tb.bullet(alt)
            tb.blank()

        if consequences:
            tb.heading("Consequences", 2)
            for con in consequences:
                tb.bullet(con)
            tb.blank()

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 8,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # Cancelled project generation
    # -------------------------------------------------------------------------

    def _generate_cancelled_projects(self) -> List[str]:
        generated = []
        for project in self.universe["projects"]:
            if project.get("status") == "cancelled":
                result = self._generate_cancelled_project(project)
                if result:
                    generated.append(result)
        return generated

    def _generate_cancelled_project(self, project: Dict) -> Optional[str]:
        name = project["name"]
        slug = f"{slugify(name)}-cancelled"
        relative_path = f"work/{slug}.md"

        if self._should_skip(relative_path):
            return None

        lead = project["lead"]
        team = project.get("team", [])
        reason = project.get("cancellation_reason", "Project deprioritized due to resource constraints")
        technologies = project.get("technologies", [])
        goal = project.get("goal", "")

        entities: List[Dict] = [
            {"name": name, "type": "project"},
            {"name": lead, "type": "person"},
        ]
        cooccurrences: List[Dict] = [{"entity_a": name, "entity_b": lead}]

        for member in team:
            entities.append({"name": member, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": member})

        for tech in technologies:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": name, "entity_b": tech})

        tb = TextBuilder()
        tb.heading(f"{name} — Cancelled")

        tb.heading("Overview", 2)
        start = MONTH_NAMES.get(project["start_month"], project["start_month"])
        end = MONTH_NAMES.get(project.get("end_month", ""), "")
        tb.paragraph(f"The {name} project was initiated in {start} with the goal to {goal}. "
                    f"The project was led by {lead} with team members {', '.join(team)}.")

        tb.heading("Cancellation Reason", 2)
        tb.paragraph(reason)

        tb.heading("Work Completed", 2)
        if technologies:
            tb.paragraph(f"Before cancellation, the team had made progress on the "
                        f"{', '.join(technologies[:2])} foundation. "
                        f"This work may be reusable in future initiatives.")

        tb.heading("Handoff", 2)
        tb.paragraph(f"Reusable components from {name} have been documented and merged into "
                    f"the shared codebase. Team members have been reassigned to higher-priority projects.")

        content = tb.build()
        entities = self._dedupe_entities(entities)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 7,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # Query generation
    # -------------------------------------------------------------------------

    def _generate_queries(self):
        """Generate eval_queries.json with ~160 queries."""
        queries: List[Dict] = []
        query_id = 0

        # --- Retrieval queries (60) ---
        # Person + technology (20)
        for person in self.universe["people"]:
            tech_skills = self._tech_skills_for(person["name"])
            if tech_skills:
                query_id += 1
                source_docs = [f"associates/{slugify(person['name'])}.md"]
                for proj in self.universe["projects"]:
                    if person["name"] in ([proj["lead"]] + proj["team"]):
                        source_docs.append(f"work/{slugify(proj['name'])}.md")

                queries.append({
                    "id": f"retrieval_{query_id:03d}",
                    "type": "retrieval",
                    "question": f"What technologies does {person['name']} work with?",
                    "expected_entities": [person["name"]] + tech_skills[:3],
                    "expected_answer_keywords": tech_skills[:3],
                    "expected_source_documents": source_docs[:3],
                    "min_relevant_chunks": 2,
                    "time_context": None,
                    "hops_required": 1,
                })
                if query_id >= 20:
                    break

        # Project + people (12)
        for proj in self.universe["projects"]:
            query_id += 1
            team = [proj["lead"]] + proj["team"][:2]
            queries.append({
                "id": f"retrieval_{query_id:03d}",
                "type": "retrieval",
                "question": f"Who is working on the {proj['name']} project?",
                "expected_entities": [proj["name"]] + team,
                "expected_answer_keywords": team,
                "expected_source_documents": [f"work/{slugify(proj['name'])}.md"],
                "min_relevant_chunks": 1,
                "time_context": None,
                "hops_required": 1,
            })
            if query_id >= 32:
                break

        # Technology + project (12)
        for proj in self.universe["projects"]:
            if proj["technologies"] and proj.get("status") != "cancelled":
                query_id += 1
                queries.append({
                    "id": f"retrieval_{query_id:03d}",
                    "type": "retrieval",
                    "question": f"What technology stack is used in the {proj['name']} project?",
                    "expected_entities": [proj["name"]] + proj["technologies"][:3],
                    "expected_answer_keywords": proj["technologies"][:3],
                    "expected_source_documents": [f"work/{slugify(proj['name'])}.md"],
                    "min_relevant_chunks": 1,
                    "time_context": None,
                    "hops_required": 1,
                })
                if query_id >= 44:
                    break

        # Person role/team (16)
        for person in self.universe["people"]:
            query_id += 1
            queries.append({
                "id": f"retrieval_{query_id:03d}",
                "type": "retrieval",
                "question": f"What is {person['name']}'s role and team?",
                "expected_entities": [person["name"]],
                "expected_answer_keywords": [person["role"], person["team"]],
                "expected_source_documents": [f"associates/{slugify(person['name'])}.md"],
                "min_relevant_chunks": 1,
                "time_context": None,
                "hops_required": 1,
            })
            if query_id >= 60:
                break

        # --- Temporal queries (25) ---
        temporal_id = 0
        for month_entry in self.universe["timeline"]:
            month = month_entry["month"]
            for event in month_entry["events"]:
                if event["type"] in ("project_kickoff", "project_completed"):
                    temporal_id += 1
                    participants = event.get("participants", [])
                    detail = event.get("detail", event.get("project", ""))
                    ts = month_to_timestamp(month)
                    queries.append({
                        "id": f"temporal_{temporal_id:03d}",
                        "type": "temporal",
                        "question": f"What happened with {detail} in {MONTH_NAMES.get(month, month)}?",
                        "expected_entities": participants[:3] if participants else [detail],
                        "expected_answer_keywords": [detail] + participants[:2],
                        "expected_source_documents": [],
                        "min_relevant_chunks": 1,
                        "time_context": {"start": ts - 86400 * 15, "end": ts + 86400 * 15},
                        "hops_required": 1,
                    })
                elif event["type"] == "decision":
                    ref = event.get("ref")
                    if ref:
                        decision = self.registry.get_decision(ref)
                        if decision:
                            temporal_id += 1
                            ts = month_to_timestamp(month)
                            queries.append({
                                "id": f"temporal_{temporal_id:03d}",
                                "type": "temporal",
                                "question": f"What was decided regarding {decision['detail'].lower()} in {MONTH_NAMES.get(month, month)}?",
                                "expected_entities": decision["participants"][:2],
                                "expected_answer_keywords": [decision["detail"]] + decision["participants"][:2],
                                "expected_source_documents": [],
                                "min_relevant_chunks": 1,
                                "time_context": {"start": ts - 86400 * 15, "end": ts + 86400 * 15},
                                "hops_required": 1,
                            })
                elif event["type"] == "incident":
                    ref = event.get("ref")
                    if ref:
                        incident = self.registry.get_incident(ref)
                        if incident:
                            temporal_id += 1
                            ts = month_to_timestamp(month)
                            queries.append({
                                "id": f"temporal_{temporal_id:03d}",
                                "type": "temporal",
                                "question": f"What caused the {incident['detail'].split('—')[0].strip().lower()} incident in {MONTH_NAMES.get(month, month)}?",
                                "expected_entities": incident["participants"][:2] + incident.get("affected_tech", [])[:2],
                                "expected_answer_keywords": [incident.get("root_cause", incident["detail"])[:80]],
                                "expected_source_documents": [],
                                "min_relevant_chunks": 1,
                                "time_context": {"start": ts - 86400 * 15, "end": ts + 86400 * 15},
                                "hops_required": 1,
                            })
                if temporal_id >= 25:
                    break
            if temporal_id >= 25:
                break

        # --- Multi-hop queries (25) ---
        multihop_id = 0

        # Person -> Project -> Technology (12)
        for person in self.universe["people"]:
            for proj in self.universe["projects"]:
                if person["name"] in ([proj["lead"]] + proj["team"]):
                    multihop_id += 1
                    queries.append({
                        "id": f"multihop_{multihop_id:03d}",
                        "type": "multi_hop",
                        "question": f"What technologies is {person['name']} exposed to through their projects?",
                        "expected_entities": [person["name"], proj["name"]] + proj["technologies"][:2],
                        "expected_answer_keywords": proj["technologies"][:3],
                        "expected_source_documents": [
                            f"associates/{slugify(person['name'])}.md",
                            f"work/{slugify(proj['name'])}.md",
                        ],
                        "min_relevant_chunks": 2,
                        "time_context": None,
                        "hops_required": 2,
                    })
                    break
            if multihop_id >= 12:
                break

        # Person -> Collaborator -> Technology (13)
        for proj in self.universe["projects"]:
            all_members = [proj["lead"]] + proj["team"]
            if len(all_members) >= 2:
                multihop_id += 1
                p1, p2 = all_members[0], all_members[1]
                p2_data = self.people_by_name.get(p2, {})
                p2_tech = [s for s in p2_data.get("skills", []) if s in self.universe["technologies"]]

                queries.append({
                    "id": f"multihop_{multihop_id:03d}",
                    "type": "multi_hop",
                    "question": f"What skills does {p1}'s collaborator {p2} bring to {proj['name']}?",
                    "expected_entities": [p1, p2, proj["name"]] + p2_tech[:2],
                    "expected_answer_keywords": p2_tech[:3] + [p2],
                    "expected_source_documents": [
                        f"work/{slugify(proj['name'])}.md",
                        f"associates/{slugify(p2)}.md",
                    ],
                    "min_relevant_chunks": 2,
                    "time_context": None,
                    "hops_required": 2,
                })
                if multihop_id >= 25:
                    break

        # --- Filler rejection queries (15) ---
        filler_templates = [
            "What technical decisions were made at the team lunch?",
            "What architecture changes were discussed at the holiday party?",
            "What deployment issues came up during office hours?",
            "What database problems were discussed at the book club?",
            "What security vulnerabilities were found at the hackathon?",
            "What production incidents happened during the team offsite planning?",
            "What API changes were decided at the new year kickoff?",
            "What infrastructure problems were discussed at the intern welcome?",
            "What code reviews happened at the design review social?",
            "What testing strategy was discussed at the team lunch in March?",
            "What CI/CD changes were planned at the frontend office hours?",
            "What monitoring alerts were discussed at the backend office hours?",
            "What service mesh issues were raised at the data platform office hours?",
            "What payment bugs were found during the spring hackathon?",
            "What scaling issues were discussed at the infrastructure office hours?",
        ]
        for i, question in enumerate(filler_templates):
            queries.append({
                "id": f"filler_rejection_{i+1:03d}",
                "type": "filler_rejection",
                "question": question,
                "expected_entities": [],
                "expected_answer_keywords": [],
                "expected_source_documents": [],
                "min_relevant_chunks": 0,
                "time_context": None,
                "hops_required": 0,
            })

        # --- Cross-document chain queries (10) ---
        chain_queries = []

        # Incident -> ADR chains
        for adr in self.universe.get("adrs", []):
            prompted_by = adr.get("prompted_by") or ""
            if prompted_by.startswith("incident_"):
                incident = self.registry.get_incident(prompted_by)
                if incident:
                    chain_queries.append({
                        "id": f"cross_document_chain_{len(chain_queries)+1:03d}",
                        "type": "cross_document_chain",
                        "question": f"How did the {incident['detail'].split('—')[0].strip().lower()} incident lead to architectural changes?",
                        "expected_entities": [adr["author"]] + incident.get("affected_tech", [])[:2],
                        "expected_answer_keywords": [adr["title"], incident["detail"].split("—")[0].strip()],
                        "expected_source_documents": [
                            f"meetings/{slugify(incident['detail'].split('—')[0].strip())}-postmortem.md",
                            f"work/adr-{adr['number']:03d}-{slugify(adr['title'])}.md",
                        ],
                        "min_relevant_chunks": 2,
                        "time_context": None,
                        "hops_required": 2,
                    })
            if len(chain_queries) >= 4:
                break

        # Decision -> ADR chains
        for adr in self.universe.get("adrs", []):
            prompted_by = adr.get("prompted_by") or ""
            if prompted_by.startswith("decision_"):
                decision = self.registry.get_decision(prompted_by)
                if decision:
                    chain_queries.append({
                        "id": f"cross_document_chain_{len(chain_queries)+1:03d}",
                        "type": "cross_document_chain",
                        "question": f"What was the formal rationale behind {decision['detail'].lower()}?",
                        "expected_entities": [adr["author"]] + decision["participants"][:1],
                        "expected_answer_keywords": [adr["title"], decision["detail"]],
                        "expected_source_documents": [
                            f"work/adr-{adr['number']:03d}-{slugify(adr['title'])}.md",
                        ],
                        "min_relevant_chunks": 1,
                        "time_context": None,
                        "hops_required": 2,
                    })
            if len(chain_queries) >= 8:
                break

        # Person mentorship chains (1:1 -> associate profile)
        mentor_pairs = []
        for person in self.universe["people"]:
            if person.get("mentors"):
                mentor_pairs.append((person.get("mentors"), person["name"]))
        for mentor, mentee in mentor_pairs[:2]:
            chain_queries.append({
                "id": f"cross_document_chain_{len(chain_queries)+1:03d}",
                "type": "cross_document_chain",
                "question": f"How has {mentor}'s mentorship of {mentee} been reflected across meetings and profiles?",
                "expected_entities": [mentor, mentee],
                "expected_answer_keywords": [mentor, mentee],
                "expected_source_documents": [
                    f"associates/{slugify(mentee)}.md",
                    f"associates/{slugify(mentor)}.md",
                ],
                "min_relevant_chunks": 2,
                "time_context": None,
                "hops_required": 2,
            })
            if len(chain_queries) >= 10:
                break

        queries.extend(chain_queries[:10])

        # Write queries
        queries_path = CORPUS_DIR / "eval_queries.json"
        queries_path.write_text(json.dumps(queries, indent=4) + "\n")

        # Print summary
        from collections import Counter
        counts = Counter(q["type"] for q in queries)
        print(f"Generated {len(queries)} evaluation queries:")
        for qtype, count in sorted(counts.items()):
            print(f"  {qtype}: {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Cortex evaluation corpus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = CorpusGenerator(seed=args.seed)
    generator.generate_all()


if __name__ == "__main__":
    main()
