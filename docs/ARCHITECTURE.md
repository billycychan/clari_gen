# ClariGen System Architecture

This document outlines the architecture of ClariGen, detailing the interaction flow and the state transitions of a query.

## 1. System Interaction Flow (Sequence Diagram)

This sequence diagram illustrates exactly how a user query is processed, showing the interactions between the User, the Orchestrator (Ambiguity Pipeline), and the Model Clients.

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Frontend/API
    participant Orch as Ambiguity Pipeline<br>(Orchestrator)
    participant Small as Small LLM<br>(8B - Detection)
    participant Large as Large LLM<br>(70B - Clarification)

    User->>Frontend: Submit Query
    Frontend->>Orch: process_query(text)
    activate Orch
    
    note right of Orch: Status: CHECKING_AMBIGUITY
    Orch->>Small: detect_binary_ambiguity(query)
    Small-->>Orch: is_ambiguous (bool)

    alt Query is NOT Ambiguous
        Orch-->>Frontend: Return Original Query
        Frontend-->>User: Display Results
    else Query IS Ambiguous
        note right of Orch: Status: AMBIGUOUS
        Orch->>Large: generate_clarification(query)
        Large-->>Orch: breakdown & question
        
        note right of Orch: Status: AWAITING_CLARIFICATION
        Orch-->>Frontend: Return Clarification Question
        deactivate Orch
        Frontend-->>User: Ask Clarification Question
        
        User->>Frontend: Provide Answer
        Frontend->>Orch: continue_with_clarification(answer)
        activate Orch
        
        note right of Orch: Status: REFORMULATING
        Orch->>Large: reformulate_query(original, answer)
        Large-->>Orch: reformulated_query
        
        note right of Orch: Status: AWAITING_CONFIRMATION
        Orch-->>Frontend: Return Reformulated Query
        deactivate Orch
        Frontend-->>User: Request Confirmation
        
        User->>Frontend: Confirm (Yes/No/Edit)
        Frontend->>Orch: confirm_reformulation()
        activate Orch
        
        note right of Orch: Status: COMPLETED
        Orch-->>Frontend: Final Query Object
        deactivate Orch
        Frontend-->>User: Process Final Query
    end
```

## 2. Query Lifecycle (State Diagram)

The `Query` object transitions through several states managed by the pipeline.

```mermaid
stateDiagram-v2
    [*] --> CHECKING_AMBIGUITY
    
    CHECKING_AMBIGUITY --> COMPLETED: Not Ambiguous
    CHECKING_AMBIGUITY --> AMBIGUOUS: Detected Ambiguity
    
    AMBIGUOUS --> AWAITING_CLARIFICATION: Question Generated
    
    AWAITING_CLARIFICATION --> REFORMULATING: User Answer Received
    
    REFORMULATING --> AWAITING_CONFIRMATION: Reformulation Done
    
    AWAITING_CONFIRMATION --> COMPLETED: User Confirmed
    
    COMPLETED --> [*]
```

## 3. Key Components

*   **AmbiguityPipeline**: The central controller defined in `ambiguity_pipeline.py`. It manages the `Query` object and coordinates calls to the models.
*   **SmallModelClient**: Specialized for speed. It uses the 8B model solely for the binary decision: "Is this ambiguous?".
*   **LargeModelClient**: Specialized for reasoning. It uses the 70B model to understand *why* it's ambiguous, generate a precise question, and rewrite the query based on the answer.
