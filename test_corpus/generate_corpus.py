#!/usr/bin/env python3
"""
Deterministic corpus generator for Cortex evaluation framework.

Reads universe.json and produces meeting transcripts, associate profiles,
and project summaries with corresponding .expected.json ground truth files.

Usage:
    python test_corpus/generate_corpus.py [--seed 42]

Existing hand-written documents are never overwritten.
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

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
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def slugify(name: str) -> str:
    """Convert a name to a filename slug."""
    return name.lower().replace(" ", "-").replace("/", "-").replace("'", "")


def month_to_timestamp(month_str: str, day: int = 15) -> int:
    """Convert '2024-08' + day to Unix timestamp."""
    dt = datetime(int(month_str[:4]), int(month_str[5:7]), day, 10, 0, 0)
    return int(dt.timestamp())


class CorpusGenerator:
    def __init__(self, seed: int = 42):
        with open(UNIVERSE_PATH) as f:
            self.universe = json.load(f)
        self.rng = random.Random(seed)
        self.people_by_name = {p["name"]: p for p in self.universe["people"]}
        self.projects_by_name = {p["name"]: p for p in self.universe["projects"]}
        self.generated_queries: List[Dict] = []

    def generate_all(self):
        """Generate the full corpus: meetings, associates, projects."""
        meetings = self._generate_meetings()
        associates = self._generate_associates()
        projects = self._generate_projects()

        print(f"Generated {len(meetings)} meetings, {len(associates)} associates, "
              f"{len(projects)} project summaries")
        print(f"Total: {len(meetings) + len(associates) + len(projects)} documents "
              f"(+ 5 existing = {len(meetings) + len(associates) + len(projects) + 5} total)")

        # Generate queries from the produced documents
        self._generate_queries()

    def _should_skip(self, relative_path: str) -> bool:
        """Check if this file already exists as a hand-written document."""
        return relative_path in EXISTING_FILES

    def _write_doc(self, relative_path: str, content: str, expected: Dict):
        """Write a document and its .expected.json."""
        doc_path = CORPUS_DIR / relative_path
        expected_path = doc_path.with_suffix(".expected.json")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(content)
        expected_path.write_text(json.dumps(expected, indent=4) + "\n")

    # -------------------------------------------------------------------------
    # Meeting generation
    # -------------------------------------------------------------------------

    def _generate_meetings(self) -> List[str]:
        """Generate meeting transcripts from timeline events."""
        generated = []
        meeting_counter = 0

        for month_entry in self.universe["timeline"]:
            month = month_entry["month"]
            events = month_entry["events"]

            for event in events:
                if event["type"] == "meeting":
                    meeting_counter += 1
                    result = self._generate_single_meeting(event, month, meeting_counter)
                    if result:
                        generated.append(result)

                elif event["type"] == "project_kickoff":
                    meeting_counter += 1
                    result = self._generate_kickoff_meeting(event, month, meeting_counter)
                    if result:
                        generated.append(result)

                elif event["type"] == "incident":
                    meeting_counter += 1
                    result = self._generate_incident_postmortem(event, month, meeting_counter)
                    if result:
                        generated.append(result)

        # Generate additional standup meetings to fill out the corpus
        for month_entry in self.universe["timeline"]:
            month = month_entry["month"]
            for week in range(1, 5):
                meeting_counter += 1
                result = self._generate_standup(month, week, meeting_counter)
                if result:
                    generated.append(result)

        return generated

    def _generate_single_meeting(self, event: Dict, month: str, idx: int) -> Optional[str]:
        """Generate a topic-focused meeting transcript."""
        topic = event["topic"]
        participants = event["participants"]
        slug = slugify(topic)
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        entities = []
        cooccurrences = []

        # Add participant entities
        for name in participants:
            entities.append({"name": name, "type": "person"})

        # Find technologies mentioned by participants
        mentioned_tech = set()
        for name in participants:
            person = self.people_by_name.get(name, {})
            skills = person.get("skills", [])
            # Pick 1-3 skills that are actual technologies
            tech_skills = [s for s in skills if s in self.universe["technologies"]]
            chosen = self.rng.sample(tech_skills, min(len(tech_skills), self.rng.randint(1, 3)))
            mentioned_tech.update(chosen)
            for tech in chosen:
                cooccurrences.append({"entity_a": name, "entity_b": tech})

        for tech in mentioned_tech:
            entities.append({"name": tech, "type": "technology"})

        # Find related projects
        mentioned_projects = set()
        for name in participants:
            for proj in self.universe["projects"]:
                if name in ([proj["lead"]] + proj["team"]):
                    if self.rng.random() < 0.5:
                        mentioned_projects.add(proj["name"])
                        cooccurrences.append({"entity_a": name, "entity_b": proj["name"]})

        for proj_name in mentioned_projects:
            entities.append({"name": proj_name, "type": "project"})

        # Build transcript
        lines = [f"# {topic.title()} — {date_str}\n"]
        lines.append(f"## Attendees\n{', '.join(participants)}\n")

        # Discussion sections
        sections = self._generate_discussion_sections(topic, participants, mentioned_tech, mentioned_projects)
        lines.extend(sections)

        # Action items
        lines.append("\n## Action Items\n")
        for p in participants[:3]:
            person = self.people_by_name.get(p, {})
            action_tech = self.rng.choice(list(mentioned_tech)) if mentioned_tech else "the system"
            lines.append(f"- {p} will follow up on {action_tech} integration by end of week")

        content = "\n".join(lines)

        # Deduplicate entities
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e["name"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        expected = {
            "entities": unique_entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": max(3, len(participants)),
            "expected_chunks_max": max(8, len(participants) * 3),
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_discussion_sections(
        self, topic: str, participants: List[str],
        technologies: set, projects: set
    ) -> List[str]:
        """Generate realistic discussion sections for a meeting."""
        lines = []
        lines.append(f"\n## Discussion\n")

        discussion_templates = [
            "I've been looking into {tech} and I think it could solve our {topic} challenges. The performance benchmarks look promising.",
            "From my perspective, we should prioritize {tech} integration. The current approach has scalability concerns.",
            "I ran some tests with {tech} last week. The results were positive — about 40% improvement in throughput.",
            "We need to consider the trade-offs here. {tech} has a steeper learning curve but better long-term maintainability.",
            "I spoke with the {tech} team and they're willing to provide support during the rollout.",
            "The documentation for {tech} is solid. I've drafted an RFC that outlines the migration path.",
            "I have concerns about {tech} compatibility with our existing infrastructure. We should run a proof of concept first.",
            "Based on our metrics, switching to {tech} would reduce latency by approximately 30%.",
        ]

        tech_list = list(technologies)
        proj_list = list(projects)

        for i, participant in enumerate(participants):
            template = self.rng.choice(discussion_templates)
            tech = tech_list[i % len(tech_list)] if tech_list else "the new framework"
            line = template.format(tech=tech, topic=topic)
            lines.append(f"**{participant}:** {line}\n")

            # Sometimes mention a project
            if proj_list and self.rng.random() < 0.4:
                proj = self.rng.choice(proj_list)
                lines.append(f"**{participant}:** This also affects the {proj} project timeline.\n")

        # Decisions section
        lines.append("\n## Decisions\n")
        if tech_list:
            decider = participants[0]
            tech = self.rng.choice(tech_list)
            lines.append(f"**{decider}:** Based on this discussion, we'll proceed with {tech} for the next phase.\n")

        return lines

    def _generate_kickoff_meeting(self, event: Dict, month: str, idx: int) -> Optional[str]:
        """Generate a project kickoff meeting."""
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

        entities = [{"name": project_name, "type": "project"}]
        cooccurrences = []

        for name in participants:
            entities.append({"name": name, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": project_name})

        for tech in proj_tech[:4]:
            entities.append({"name": tech, "type": "technology"})
            # Lead person co-occurs with project tech
            if participants:
                cooccurrences.append({"entity_a": participants[0], "entity_b": tech})

        lines = [f"# {project_name} — Kickoff Meeting — {date_str}\n"]
        lines.append(f"## Attendees\n{', '.join(participants)}\n")
        lines.append(f"## Project Overview\n")

        if participants:
            lines.append(f"**{participants[0]}:** Welcome everyone. Today we're kicking off "
                        f"the {project_name} project. Let me walk through the goals and timeline.\n")

        lines.append(f"## Technical Approach\n")
        for i, tech in enumerate(proj_tech[:4]):
            speaker = participants[i % len(participants)]
            lines.append(f"**{speaker}:** We'll be using {tech} for this project. "
                        f"I've evaluated alternatives and this gives us the best foundation.\n")

        lines.append(f"## Timeline\n")
        lines.append(f"- Phase 1: Architecture and design ({MONTH_NAMES.get(month, month)})\n")
        lines.append(f"- Phase 2: Core implementation (following month)\n")
        lines.append(f"- Phase 3: Testing and rollout (month after)\n")

        lines.append(f"\n## Action Items\n")
        for p in participants[:3]:
            lines.append(f"- {p} will draft their section of the technical design doc by Friday\n")

        content = "\n".join(lines)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 4,
            "expected_chunks_max": 10,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_incident_postmortem(self, event: Dict, month: str, idx: int) -> Optional[str]:
        """Generate an incident postmortem meeting."""
        detail = event["detail"]
        participants = event["participants"]
        slug = slugify(detail.split("—")[0].strip()) if "—" in detail else f"incident-{idx}"
        day = self.rng.randint(1, 28)
        date_str = f"{month}-{day:02d}"
        relative_path = f"meetings/{slug}-postmortem.md"

        if self._should_skip(relative_path):
            return None

        entities = []
        cooccurrences = []

        for name in participants:
            entities.append({"name": name, "type": "person"})

        # Extract technologies from participants' skills
        incident_tech = set()
        for name in participants:
            person = self.people_by_name.get(name, {})
            tech_skills = [s for s in person.get("skills", []) if s in self.universe["technologies"]]
            chosen = self.rng.sample(tech_skills, min(len(tech_skills), 2))
            incident_tech.update(chosen)
            for tech in chosen:
                cooccurrences.append({"entity_a": name, "entity_b": tech})

        for tech in incident_tech:
            entities.append({"name": tech, "type": "technology"})

        lines = [f"# Incident Postmortem — {detail} — {date_str}\n"]
        lines.append(f"## Attendees\n{', '.join(participants)}\n")
        lines.append(f"## Incident Summary\n")
        lines.append(f"{detail}. The incident was detected by automated monitoring and "
                    f"the team responded within 5 minutes.\n")

        lines.append(f"\n## Root Cause Analysis\n")
        if participants:
            tech = self.rng.choice(list(incident_tech)) if incident_tech else "the service"
            lines.append(f"**{participants[0]}:** The root cause was a configuration issue "
                        f"in {tech}. A recent deployment changed the connection pool settings "
                        f"without updating the timeout values.\n")

        lines.append(f"\n## Resolution\n")
        if len(participants) > 1:
            lines.append(f"**{participants[1]}:** We rolled back the configuration change and "
                        f"added monitoring alerts to catch similar issues in the future.\n")

        lines.append(f"\n## Action Items\n")
        for p in participants[:2]:
            lines.append(f"- {p} will implement additional safeguards to prevent recurrence\n")

        content = "\n".join(lines)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 3,
            "expected_chunks_max": 8,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    def _generate_standup(self, month: str, week: int, idx: int) -> Optional[str]:
        """Generate a weekly standup meeting."""
        day = min(week * 7, 28)
        date_str = f"{month}-{day:02d}"
        slug = f"standup-{date_str}"
        relative_path = f"meetings/{slug}.md"

        if self._should_skip(relative_path):
            return None

        # Pick 3-5 participants for this standup
        all_engineers = [p for p in self.universe["people"]
                        if p["role"] not in ("Product Manager", "UX Designer",
                                             "Technical Writer", "Engineering Director")]
        participants = self.rng.sample(all_engineers, min(len(all_engineers), self.rng.randint(3, 5)))
        participant_names = [p["name"] for p in participants]

        entities = []
        cooccurrences = []

        for p in participants:
            entities.append({"name": p["name"], "type": "person"})

        lines = [f"# Weekly Standup — {date_str}\n"]

        # Each person gives an update
        for p in participants:
            tech_skills = [s for s in p["skills"] if s in self.universe["technologies"]]
            if tech_skills:
                tech = self.rng.choice(tech_skills)
                entities.append({"name": tech, "type": "technology"})
                cooccurrences.append({"entity_a": p["name"], "entity_b": tech})

                update_templates = [
                    f"Yesterday I worked on the {tech} integration. Made good progress on the core implementation. Today I'll continue with testing.",
                    f"I finished the {tech} configuration updates. No blockers. Moving on to documentation today.",
                    f"I've been debugging an issue with {tech}. Found the root cause — it was a version mismatch. Fixed now.",
                    f"The {tech} migration is about 70% complete. Should be done by end of week.",
                ]
                update = self.rng.choice(update_templates)
            else:
                update = "Working on code reviews and documentation updates. No blockers."

            lines.append(f"**{p['name']}:** {update}\n")

        # Brief filler at end
        lines.append(f"\n**{participant_names[0]}:** Great updates everyone. See you next week.\n")

        content = "\n".join(lines)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": len(participants),
            "expected_chunks_max": len(participants) * 2,
            "expected_filler_chunks_min": 1,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # Associate profile generation
    # -------------------------------------------------------------------------

    def _generate_associates(self) -> List[str]:
        """Generate associate profiles for people not already covered."""
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
        """Generate a structured associate profile."""
        name = person["name"]
        slug = slugify(name)
        relative_path = f"associates/{slug}.md"

        entities = [{"name": name, "type": "person"}]
        cooccurrences = []

        # Add organization
        entities.append({"name": "TechCorp", "type": "organization"})
        cooccurrences.append({"entity_a": name, "entity_b": "TechCorp"})

        if person.get("team"):
            entities.append({"name": person["team"], "type": "organization"})
            cooccurrences.append({"entity_a": name, "entity_b": person["team"]})

        # Add technologies from skills
        tech_skills = [s for s in person["skills"] if s in self.universe["technologies"]]
        for tech in tech_skills:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": name, "entity_b": tech})

        # Find projects this person is on
        person_projects = []
        for proj in self.universe["projects"]:
            if name in ([proj["lead"]] + proj["team"]):
                person_projects.append(proj)
                entities.append({"name": proj["name"], "type": "project"})
                cooccurrences.append({"entity_a": name, "entity_b": proj["name"]})

        # Find collaborators
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
        lines = [f"# {name} — {person['role']}\n"]

        lines.append("## Overview\n")
        lines.append(f"{name} is a {person['role']} at TechCorp who joined in "
                    f"{person['joined']}. They work on the {person['team']} team.\n")

        lines.append("\n## Skills & Expertise\n")
        if tech_skills:
            lines.append(f"{name} specializes in {', '.join(tech_skills[:3])}. ")
            if len(tech_skills) > 3:
                lines.append(f"They also have experience with {', '.join(tech_skills[3:])}. ")
            lines.append("\n")

        if person_projects:
            lines.append("\n## Current Projects\n")
            for proj in person_projects:
                role_str = "leading" if proj["lead"] == name else "contributing to"
                lines.append(f"{name} is currently {role_str} the {proj['name']} project")
                if proj["technologies"]:
                    lines.append(f", which uses {', '.join(proj['technologies'][:3])}")
                lines.append(".\n")

        if mentioned_collabs:
            lines.append("\n## Key Collaborators\n")
            lines.append(f"{name} works closely with {', '.join(mentioned_collabs)}.\n")

        lines.append("\n## Working Style\n")
        styles = [
            f"{name} prefers clear documentation and thorough code reviews.",
            f"{name} is known for writing detailed technical specs before implementation.",
            f"{name} favors iterative development with frequent check-ins.",
            f"{name} values pair programming and knowledge sharing sessions.",
        ]
        lines.append(f"{self.rng.choice(styles)}\n")

        content = "\n".join(lines)

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
    # Project summary generation
    # -------------------------------------------------------------------------

    def _generate_projects(self) -> List[str]:
        """Generate project summaries."""
        generated = []

        for project in self.universe["projects"]:
            slug = slugify(project["name"])
            relative_path = f"work/{slug}.md"

            if self._should_skip(relative_path):
                continue

            result = self._generate_project_summary(project)
            if result:
                generated.append(result)

        return generated

    def _generate_project_summary(self, project: Dict) -> Optional[str]:
        """Generate a structured project summary."""
        name = project["name"]
        slug = slugify(name)
        relative_path = f"work/{slug}.md"

        entities = [{"name": name, "type": "project"}]
        cooccurrences = []

        # People
        lead = project["lead"]
        entities.append({"name": lead, "type": "person"})
        cooccurrences.append({"entity_a": name, "entity_b": lead})

        for member in project["team"]:
            entities.append({"name": member, "type": "person"})
            cooccurrences.append({"entity_a": name, "entity_b": member})

        # Technologies
        for tech in project["technologies"]:
            entities.append({"name": tech, "type": "technology"})
            cooccurrences.append({"entity_a": name, "entity_b": tech})

        # Build summary
        lines = [f"# {name} — Project Summary\n"]

        lines.append("## Project Overview\n")
        team_str = ", ".join(project["team"][:3])
        lines.append(f"The {name} project was initiated in {MONTH_NAMES.get(project['start_month'], project['start_month'])} "
                    f"and is led by {lead}. The core team includes {team_str}.\n")

        lines.append("\n## Technical Architecture\n")
        if project["technologies"]:
            tech_str = ", ".join(project["technologies"][:4])
            lines.append(f"The technical stack includes {tech_str}. ")

            # Add some detail about technology choices
            for i, tech in enumerate(project["technologies"][:3]):
                reasons = [
                    f"{tech} was chosen for its strong community support and proven track record in production.",
                    f"The team selected {tech} after evaluating several alternatives, based on performance and developer experience.",
                    f"{tech} integrates well with the existing infrastructure and reduces operational complexity.",
                ]
                lines.append(f"{self.rng.choice(reasons)} ")
            lines.append("\n")

        lines.append("\n## Key Decisions\n")
        decisions = [
            f"Adopted {project['technologies'][0] if project['technologies'] else 'agile methodology'} as the primary framework",
            f"Implemented CI/CD pipeline using GitHub Actions for automated testing and deployment",
            f"Established weekly sync meetings between {lead} and stakeholders",
        ]
        for i, d in enumerate(decisions, 1):
            lines.append(f"{i}. {d}\n")

        lines.append("\n## Timeline\n")
        start = MONTH_NAMES.get(project["start_month"], project["start_month"])
        lines.append(f"- {start}: Project kickoff, initial architecture design\n")
        if project["end_month"]:
            end = MONTH_NAMES.get(project["end_month"], project["end_month"])
            lines.append(f"- {end}: Project completion and production rollout\n")
        else:
            lines.append(f"- Ongoing: Active development and iteration\n")

        lines.append(f"\n## Status\n")
        status_map = {
            "completed": "The project has been successfully completed and is in production.",
            "in_progress": "The project is actively under development.",
            "planning": "The project is in the planning phase with design work underway.",
        }
        lines.append(f"{status_map.get(project['status'], 'Status unknown.')}\n")

        content = "\n".join(lines)

        expected = {
            "entities": entities,
            "cooccurrences": cooccurrences,
            "expected_chunks_min": 4,
            "expected_chunks_max": 10,
            "expected_filler_chunks": 0,
        }

        self._write_doc(relative_path, content, expected)
        return relative_path

    # -------------------------------------------------------------------------
    # Query generation
    # -------------------------------------------------------------------------

    def _generate_queries(self):
        """Generate eval_queries.json from the universe data."""
        queries = []
        query_id = 0

        # --- Retrieval queries (50) ---
        # Person + technology queries
        for person in self.universe["people"]:
            tech_skills = [s for s in person["skills"] if s in self.universe["technologies"]]
            if tech_skills:
                query_id += 1
                # Find source documents for this person
                source_docs = []
                for proj in self.universe["projects"]:
                    if person["name"] in ([proj["lead"]] + proj["team"]):
                        source_docs.append(f"work/{slugify(proj['name'])}.md")
                source_docs.append(f"associates/{slugify(person['name'])}.md")

                queries.append({
                    "id": f"retrieval_{query_id:02d}",
                    "type": "retrieval",
                    "question": f"What technologies does {person['name']} work with?",
                    "expected_entities": [person["name"]] + tech_skills[:3],
                    "expected_answer_keywords": tech_skills[:3],
                    "expected_source_documents": source_docs[:2],
                    "min_relevant_chunks": 2,
                    "time_context": None,
                    "hops_required": 1,
                })
                if query_id >= 15:
                    break

        # Project + people queries
        for proj in self.universe["projects"]:
            query_id += 1
            team = [proj["lead"]] + proj["team"][:2]
            queries.append({
                "id": f"retrieval_{query_id:02d}",
                "type": "retrieval",
                "question": f"Who is working on the {proj['name']} project?",
                "expected_entities": [proj["name"]] + team,
                "expected_answer_keywords": team,
                "expected_source_documents": [f"work/{slugify(proj['name'])}.md"],
                "min_relevant_chunks": 1,
                "time_context": None,
                "hops_required": 1,
            })
            if query_id >= 25:
                break

        # Technology + project queries
        for proj in self.universe["projects"]:
            if proj["technologies"]:
                query_id += 1
                queries.append({
                    "id": f"retrieval_{query_id:02d}",
                    "type": "retrieval",
                    "question": f"What technology stack is used in the {proj['name']} project?",
                    "expected_entities": [proj["name"]] + proj["technologies"][:3],
                    "expected_answer_keywords": proj["technologies"][:3],
                    "expected_source_documents": [f"work/{slugify(proj['name'])}.md"],
                    "min_relevant_chunks": 1,
                    "time_context": None,
                    "hops_required": 1,
                })
                if query_id >= 35:
                    break

        # Person role/team queries
        for person in self.universe["people"]:
            query_id += 1
            queries.append({
                "id": f"retrieval_{query_id:02d}",
                "type": "retrieval",
                "question": f"What is {person['name']}'s role and team?",
                "expected_entities": [person["name"]],
                "expected_answer_keywords": [person["role"], person["team"]],
                "expected_source_documents": [f"associates/{slugify(person['name'])}.md"],
                "min_relevant_chunks": 1,
                "time_context": None,
                "hops_required": 1,
            })
            if query_id >= 50:
                break

        # --- Temporal queries (20) ---
        temporal_id = 0
        for month_entry in self.universe["timeline"]:
            month = month_entry["month"]
            for event in month_entry["events"]:
                if event["type"] in ("project_kickoff", "project_completed", "decision"):
                    temporal_id += 1
                    participants = event.get("participants", [])
                    detail = event.get("detail", event.get("project", ""))

                    ts = month_to_timestamp(month)
                    # Query for state at that time
                    queries.append({
                        "id": f"temporal_{temporal_id:02d}",
                        "type": "temporal",
                        "question": f"What happened with {detail} in {MONTH_NAMES.get(month, month)}?",
                        "expected_entities": participants[:3] if participants else [detail],
                        "expected_answer_keywords": [detail] + participants[:2],
                        "expected_source_documents": [],
                        "min_relevant_chunks": 1,
                        "time_context": {"start": ts - 86400 * 15, "end": ts + 86400 * 15},
                        "hops_required": 1,
                    })
                    if temporal_id >= 20:
                        break
            if temporal_id >= 20:
                break

        # --- Multi-hop queries (20) ---
        multihop_id = 0

        # Person → Project → Technology (2 hops)
        for person in self.universe["people"]:
            for proj in self.universe["projects"]:
                if person["name"] in ([proj["lead"]] + proj["team"]):
                    multihop_id += 1
                    queries.append({
                        "id": f"multihop_{multihop_id:02d}",
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
                    break  # One query per person
            if multihop_id >= 10:
                break

        # Person → Collaborator → Technology (2+ hops)
        for proj in self.universe["projects"]:
            all_members = [proj["lead"]] + proj["team"]
            if len(all_members) >= 2:
                multihop_id += 1
                p1, p2 = all_members[0], all_members[1]
                p2_data = self.people_by_name.get(p2, {})
                p2_tech = [s for s in p2_data.get("skills", []) if s in self.universe["technologies"]]

                queries.append({
                    "id": f"multihop_{multihop_id:02d}",
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
                if multihop_id >= 20:
                    break

        # Write queries
        queries_path = CORPUS_DIR / "eval_queries.json"
        queries_path.write_text(json.dumps(queries, indent=4) + "\n")

        print(f"Generated {len(queries)} evaluation queries:")
        retrieval = sum(1 for q in queries if q["type"] == "retrieval")
        temporal = sum(1 for q in queries if q["type"] == "temporal")
        multihop = sum(1 for q in queries if q["type"] == "multi_hop")
        print(f"  Retrieval: {retrieval}, Temporal: {temporal}, Multi-hop: {multihop}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Cortex evaluation corpus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    args = parser.parse_args()

    generator = CorpusGenerator(seed=args.seed)
    generator.generate_all()


if __name__ == "__main__":
    main()
