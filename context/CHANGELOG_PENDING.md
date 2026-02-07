# Pending Documentation & Policy Changes (For User Review)

This file is append-only. Items are added whenever a design-doc-sync operation occurs.

Format per entry:
- CHG-#### | timestamp | actor | status (proposed|applied) | summary
  - refs: (doc paths, decision ids)
  - rationale:
  - diff_summary:
  - user_review_notes:

---

- CHG-0001 | 2026-02-06 | actor: design-doc-author | status: applied | Created high-level architecture design document
  - refs: file:design/high-level-architecture.md, DEC-0001, DEC-0002, DEC-0003, DEC-0004
  - rationale: Establishes foundational architecture for 2048 RL project including component structure, standardized interface, technical constraints, and critical invariants
  - diff_summary: New file created with project goals, component diagram, env-agent interface spec, technical constraints (PyTorch, GH200/ARM), game environment internals, and episode boundary handling requirements
  - user_review_notes: Document defines binding constraints for framework (PyTorch), hardware (GH200/aarch64), state representation (N,16,17 one-hot), and episode boundary handling. Open items remain for algorithm module structure, orchestrator details, and directory structure.

- CHG-0002 | 2026-02-06 | actor: design-doc-author | status: applied | Appended RL Algorithm Modules, Input Representations, and Experimental Design sections to high-level architecture
  - refs: file:design/high-level-architecture.md, DEC-0005, DEC-0006, DEC-0007, DEC-0008, DEC-0009
  - rationale: Addresses open items from CHG-0001 by specifying algorithm module structure, representation layer interface, algorithm tiering, and experimental matrix design
  - diff_summary: Added sections 7 (RL Algorithm Module Structure with folder convention, interface contract, tier classification), 8 (Input Representations with folder structure, interface, CNN hyperparameters), and 9 (Experimental Design with matrix structure and tuning approach). Updated section 6 open items to reflect resolved topics.
  - user_review_notes: New binding decisions logged: algorithm folder structure (DEC-0005), train/evaluate interface (DEC-0006), excluded algorithms DDPG/TD3/SAC (DEC-0007), representation interface (DEC-0008), experimental matrix structure (DEC-0009). Remaining open items: Training Orchestrator details, directory structure beyond algorithms/ and representations/.

- CHG-0003 | 2026-02-06 | actor: design-doc-author | status: applied | Updated section 9.2 Hyperparameter Tuning with Optuna strategy
  - refs: file:design/high-level-architecture.md, DEC-0009, DEC-0010, DEC-0011, DEC-0012
  - rationale: Resolves TBD tuning strategy by specifying Optuna with SQLite storage, one study per experimental combination, and MedianPruner configuration
  - diff_summary: Replaced placeholder section 9.2 with detailed Optuna specification including library choice, storage backend, study structure, hyperparameter categories, MedianPruner configuration (n_startup_trials=5, n_warmup_steps=10, interval_steps=5), and training integration pattern with trial.report/should_prune
  - user_review_notes: New binding decisions logged: Optuna with SQLite (DEC-0010), one study per combo (DEC-0011), MedianPruner config (DEC-0012). Trials per study remains TBD. Referenced existing DEC-0009 for experimental matrix structure.

- CHG-0004 | 2026-02-06 | actor: design-doc-author | status: applied | Appended Test-First Development Strategy (section 10) and Milestone 1 (section 11) to high-level architecture
  - refs: file:design/high-level-architecture.md, DEC-0002, DEC-0003, DEC-0004, DEC-0013, DEC-0014, DEC-0015, DEC-0016, DEC-0017
  - rationale: Establishes test-first development methodology for GameEnv, defines comprehensive test categories, documents game rule invariants, and defines Milestone 1 as test suite creation before implementation
  - diff_summary: Added section 10 (Test-First Development Strategy) with GameEnv interface contract, deterministic spawn injection for testing, game rules invariants table (merge-once, merge order, slide after merge, invalid move exception, spawn rules), comprehensive test categories table (16 categories), and merge order test examples. Added section 11 (Milestone 1: Test Suite Creation) with definition, deliverables, and success criteria.
  - user_review_notes: New binding decisions logged: test-first development (DEC-0013), invalid move exception (DEC-0014), merge-once rule (DEC-0015), deterministic spawn injection (DEC-0016), Milestone 1 definition (DEC-0017). Referenced existing decisions: DEC-0002 (state representation), DEC-0003 (episode boundary), DEC-0004 (spawn rules).

- CHG-0005 | 2026-02-06 | actor: design-doc-author | status: applied | Created project milestones design document
  - refs: file:design/milestones.md, DEC-0009, DEC-0010, DEC-0011, DEC-0012, DEC-0013, DEC-0017, DEC-0018, DEC-0019, DEC-0020, DEC-0021
  - rationale: Establishes complete project milestone roadmap with fail-fast approach, defining 27 milestones from test suite creation through final analysis
  - diff_summary: New file created with milestone summary table, detailed milestone descriptions (M1-M27) covering test suite, GameEnv, DQN basic, input representations, hyperparameter tuning, training orchestrator, Tier 1-4 algorithms, full experimental sweep, and final analysis/report
  - user_review_notes: New binding decisions logged: fail-fast milestone approach (DEC-0018), GameEnv success criteria (DEC-0019), algorithm milestone success criteria (DEC-0020), full experimental sweep scope (DEC-0021). Referenced existing decisions: DEC-0009 (experimental matrix), DEC-0010/0011/0012 (Optuna tuning), DEC-0013/0017 (test-first development, M1 definition).
