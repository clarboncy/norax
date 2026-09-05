# Security Policy

## Supported versions

Security fixes are applied to the latest release and the default branch.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, private memory, remote execution, filesystem access, or authentication bypasses.

Report vulnerabilities privately through GitHub Security Advisories for this repository.

- affected version or commit;
- reproduction steps;
- expected and observed behavior;
- impact assessment;
- suggested mitigation, if known.

Do not include live secrets, private user data, or destructive proof-of-concept payloads. You should receive an acknowledgment before public disclosure.

## Deployment boundaries

Norax is a privileged agent runtime. Operators are responsible for:

- binding control interfaces to trusted networks;
- configuring strong tokens for externally reachable HTTP or dashboard endpoints;
- restricting Discord users, guilds, and channels;
- limiting MCP filesystem roots and remote-node access;
- protecting `.env`, memory, logs, checkpoints, and event evidence;
- reviewing tool risk tiers and sender permissions;
- rotating credentials that appear in logs, patches, shell history, or Git history;
- running the runtime as an unprivileged user rather than root.

The default repository configuration disables Discord and contains no owner identity or credential.
