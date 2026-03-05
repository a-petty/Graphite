# Q3 Design Review — 2024-09-15

## Attendees
Sarah Chen, Marcus Johnson, Priya Patel, David Kim

## Dashboard Redesign

**Sarah Chen:** I want to start with the Dashboard Redesign project. We've been working on this for about six weeks now, and I think we're ready to move into the implementation phase. The wireframes are finalized and the stakeholders signed off last Thursday.

**Marcus Johnson:** Agreed. I've been evaluating React and Vue for the frontend framework. My recommendation is React with TypeScript. The team has more experience with it, and the component library we want to use, Shadcn UI, is React-native.

**Priya Patel:** I have concerns about the backend API. We're currently using Django REST Framework, but the new dashboard requires real-time updates. I think we should add WebSocket support through Django Channels. It integrates well with our existing authentication system.

**David Kim:** From a DevOps perspective, we need to consider the deployment pipeline. I've been setting up Kubernetes clusters on AWS for the staging environment. The current Docker setup works, but we need Helm charts for the production rollout.

## API Migration

**Sarah Chen:** Let's move on to the API Migration. We decided last month to migrate from REST to GraphQL for the internal services. Where are we on that?

**Marcus Johnson:** I've started prototyping with Apollo Server. The schema design is about 60% complete. The main challenge is the legacy payment service — it uses SOAP endpoints that we'll need to wrap.

**Priya Patel:** I spoke with the Acme Corp team about their integration timeline. They need at least three months' notice before we deprecate the v2 REST endpoints. We should keep both running in parallel until Q1 2025.

## Action Items

**Sarah Chen:** Let me summarize the action items:
- Marcus will finalize the React component architecture by next Friday
- Priya will write the WebSocket integration RFC and share it with the backend team
- David will complete the Helm chart templates and run a staging deployment test
- Everyone reviews the GraphQL schema draft that Marcus will share by Wednesday
