# graph_vlm_rag — Use Cases

This document outlines practical real-world applications for the Multi-Modal Hybrid GraphRAG framework.

---

## Why This Framework

Before diving into specific use cases, here's why this architecture fits modern document-heavy workflows:

| Feature | Benefit |
|---------|---------|
| **Multi-modal (PDFs + images + tables)** | Real-world documents have charts, diagrams, scanned pages — OCR/VLM handles them |
| **Hybrid search (dense + BM25)** | Natural language queries AND specific IDs/names/terms both work |
| **Knowledge graph** | Relationships between entities (who→works_for→company, drug→treats→disease) |
| **Adaptive context expansion** | Documents split across sections; queries need multi-chunk context |
| **Local-first (Docker + Ollama)** | Sensitive data (legal, medical, financial) stays on-prem |
| **Multi-document support** | Real workflows involve document collections, not single files |

---

## Enterprise & Business

### 1. Contract Intelligence & Legal Review

**Input:** Thousands of contracts, NDAs, MSAs, SOWs (PDF with scanned signatures, exhibits, amendments)

**What gets extracted:**
- Parties, effective dates, expiration dates
- Clauses (indemnification, IP rights, termination)
- Obligations, liabilities, penalties
- Cross-references to other documents

**Query examples:**
> *"What are all mycontracts with Acme Corp expiring before Q3 2026?"*

> *"Show me any NDAs with non-disclosure clauses that lack a non-compete"*

---

### 2. Financial Report Analysis

**Input:** 10-Ks, 10-Qs, earnings call transcripts, analyst reports (with charts, tables, footnotes)

**What gets extracted:**
- Companies, executives, board members
- Revenue figures, P/E ratios, segment data
- Subsidiaries, parent companies
- Guidance, projections, risk factors

**Query examples:**
> *"Compare Q3 revenue across all tech companies in my portfolio"*

> *"What did CEO Jane Doe say about AI investments in the last earnings call?"*

---

### 3. M&A Due Diligence

**Input:** Target company's entire document corpus (contracts, IP filings, employee lists, litigation, financial statements)

**What gets extracted:**
- IP assignments and ownership
- Key employees and their contracts
- Outstanding litigation and liabilities
- Financial guarantees and debt

**Query examples:**
> *"Show me all IP assignments from employee X to anyone outside company Y"*

> *"What's the total outstanding litigation exposure?"*

---

## Healthcare & Life Sciences

### 4. Clinical Research Assistant

**Input:** Medical journals, clinical trial reports, FDA submissions (with trial results tables, drug structure diagrams)

**What gets extracted:**
- Drugs, compounds, dosing
- Medical conditions, indications
- Trial phases, sites, investigators
- Outcomes, adverse events

**Query examples:**
> *"What are all trials testing drug X for Alzheimer's with positive Phase 2 results?"*

> *"Which institutions have run the most Phase 3 trials for oncology in 2024?"*

---

### 5. Pharmaceutical Knowledge Management

**Input:** Patent documents, research papers, regulatory filings, package inserts

**What gets extracted:**
- Compounds, molecular structures
- Target proteins, pathways
- Indications, contraindications
- Drug interactions

**Query examples:**
> *"Which compounds target protein Z and are currently in Phase 2 or later?"*

> *"What are all approved drugs with mechanism of action M?"*

---

## Research & Academia

### 6. Scientific Literature Review

**Input:** Thousands of papers in a research field (with methodology charts, result tables, citations)

**What gets extracted:**
- Authors, institutions, citations
- Methods, datasets, models
- Results, statistical significance
- Research gaps

**Query examples:**
> *"What methods have been used for multi-document summarization in 2024-2025?"*

> *"Who has published the most papers on topic X in the last 3 years?"*

---

### 7. Patent Landscape Analysis

**Input:** Patent documents, Office Action responses, prior art (with technical diagrams)

**What gets extracted:**
- Inventors, assignees, filing dates
- Claims, prior art citations
- Technical categories, classifications

**Query examples:**
> *"Find all patents citing technique X in autonomous vehicles"*

> *"What's the claim density for company Y's portfolio in AI?"*

---

## Government & Public Sector

### 8. Policy & Regulation Analysis

**Input:** Laws, regulations, policy documents, guidance letters (with legal references, amendment histories)

**What gets extracted:**
- Agencies, statutes, regulations
- Definitions, requirements
- Amendment histories, effective dates
- Jurisdictional scopes

**Query examples:**
> *"What regulations apply to AI systems in healthcare across federal agencies?"*

> *"Show me all changes to tax code section 409A in the last 5 years"*

---

## Tech & Engineering

### 9. Technical Documentation Assistant

**Input:** Engineering docs, runbooks, post-mortems, architecture diagrams (with flowcharts)

**What gets extracted:**
- Services, endpoints, dependencies
- Incidents, root causes, resolutions
- Teams, on-call rotations

**Query examples:**
> *"What services depend on database X and have had outages in the last 30 days?"*

> *"Show me the architecture for auth service with incident history"*

---

### 10. Codebase + Specs RAG

**Input:** Architecture docs, RFCs, API specs, design documents (with sequence diagrams, ERDs)

**What gets extracted:**
- Services, endpoints, schemas
- Design decisions, trade-offs
- Dependencies, call flows

**Query examples:**
> *"Show me the design rationale for the auth service refresh token flow"*

> *"What are all endpoints that call service X?"*

---

### 11. Customer Support Knowledge Base

**Input:** Product manuals, FAQs, troubleshooting guides (with error code tables, wiring diagrams)

**What gets extracted:**
- Products, models, SKUs
- Error codes, symptoms, solutions
- Workarounds, known issues

**Query examples:**
> *"How do I resolve error E1234 on product P model M?"*

> *"What's the troubleshooting flow for device D showing warning W?"*

---

## Manufacturing & Industry

### 12. Quality Management System

**Input:** SOPs, inspection reports, defect logs (with images of defects, measurement tables)

**What gets extracted:**
- Parts, assemblies, serial numbers
- Defect types, severity, root causes
- Production lines, shifts, operators

**Query examples:**
> *"What defects are most common in part X manufactured in facility Y?"*

> *"Show me the trend of defect severity over the last quarter"*

---

### 13. Real Estate Document Analysis

**Input:** Property listings, inspection reports, market data, lease agreements (with floor plans, financial tables)

**What gets extracted:**
- Properties, addresses, owners
- Sale prices, rental rates, cap rates
- Zoning, permits, inspections

**Query examples:**
> *"Show all commercial properties sold in downtown in the last year"*

> *"What multifamily buildings in district X have Cap Rate > 6%?"*

---

## High-Value Use Cases

| # | Use Case | Market Fit | Technical Depth |
|---|---------|-----------|-----------------|
| 1 | **Contract Intelligence** | LegalTech is billions; every enterprise has contracts | Multi-doc, entity relationships, temporal queries |
| 2 | **Financial Report Analysis** | Hedge funds, PE firms pay premium for alpha | Multi-modal charts/tables, structured extraction |
| 3 | **Clinical Research Assistant** | Pharma R&D is massive; AI in healthcare hot | Domain schema, trial phase tracking |

---

## Extending This Framework

To adapt these use cases to your specific domain:

1. **Update `data/domain_schema.yaml`** with domain-specific entities and relationships
2. **Tune the chunker** for document structure (layout-aware parameters)
3. **Adjust adaptive expansion** based on typical content spread
4. **Add domain-specific eval questions** for benchmarking

---

*Last updated: 2025-06-10*