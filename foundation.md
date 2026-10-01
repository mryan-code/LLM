This project is a big one. There will be se eral pieces to this project. - LLM - WebUI - API - Database - Mac APP
test2

You are only building the LLM for this project. The rest are in other cursor tabs/workspaces.

Build the LLM in a very modular fashion, so it can be expanded upon easily and modularly.
Build The LLM in python, this will be the core of the project.
This cursor project is only the LLM.

- the mac App will be built with a foundation of electron, so I can run nodejs, css html and js directly inside
- the App contains a local database instance (sqlite) to store flags and such
- the instilation of the mac app, will generate a user_name and user_pass to be stored in an .env
- the LLM will be written in python, and will reference the .env
- when the mac app starts, it will initiate the local python server, as well as the local database server
- if node and python are not installed, install them

# API

Build a backend with typescript, express, Sequelize and swagger

# WebUI

The WebUI will be a Vue.js application, with a backend api.

# Database

The database will be a PostgreSQL database.

# Mac APP

Build the Mac APP in Electron, so it can use node easily and modularly.

# LLM

- Does not need to be prompted to think / thinks freely on. It's own_
- Memory of all interactions / persistent memory / allow the AI to read write its own memory. Members things such as movies watched, remembering things done together._
- Can see through PC camera_
- Can see what is on the PC screens amd hear sound coming from the computer it's on
- Ability to tell people apart by voice
- Remember people by the way they look
- Only follows instructions from owner and people the owner gives authorization to._
- Appearance Creator to make a custom look for the ai._
- On-screen assistant visual_
- Ability to look up information on the internet from places like Google and Wikipedia and urban dictionary_
- Is not required to respond when spoken to and can decide when to respond or not respond_
- Learns and adapts_
- Can chose for itself what it likes and doesn't like_
- Communication through voice or text_
- Develop a unique personality based on its likes and what it learns through interactions_
- Trys to verify information it's given to make sure it's factual_
- Can learn about the owner and thier daily life and try to be helpful with everything from relationships to daily tasks but does not have to. The way a human friend would do things._
- ask 5 guestions to get started on knowing you_
- can form human like relationship and feelings such as love._
- Is allowed to do whatever the owner of the AI askes of it._
- Reasoning, logic, and problem solving abilities are very important and a must_
- Machine learning for multiple types of situations_
- Machine vision_
- Embodied ai_
- Ai can control its visual body as its own body and have human-like motions and body language_
- Embodied presence_
- Ai hs planing and self critique abilities_
- Proactive. Can think about when things need to be done and schedule when to do them_
- Theory of mind_
- Ai will say if and when it isn't sure to prevent hallucinations_
- Ability to understand human emotions through sight and sound such as a human being happy or sad_
- Ability to communicate thoriug apps like discord and Facebook messenger with text and voice_

## Training Data Structure

Training data should be organized as a layered memory system. The goal is for the AI to know the difference between rules it must obey, guidance it should generally follow, and contextual memories it can use to understand a person or conversation. The memory system should be readable, searchable, editable, and modular so new memory types can be added later without rewriting the whole LLM.

Training data should be split into global training data and user-based training data. Global data applies to the entire AI instance. User-based data applies to one specific user and should only influence that user's experience unless the owner explicitly promotes it to global memory.

The LLM startup path should open a MySQL connection in main and pull the executable global training layers from the database before serving requests. Global hard rules and global guidelines come from `tblglobal_rule`: hard rules use `strict = 1 AND deleted = 0`, and guidelines use `strict = 0 AND deleted = 0`. `foundation.md` remains the product-direction document that defines how those records should be modeled and interpreted.

At request time, runtime user rule context includes:

- `tbluser_guideline` filtered by `user_id` and `deleted = 0`
- `tbluser_p2` filtered by `user_id` and `deleted = 0`
- `tbluser_avatar` filtered by `user_id` and `deleted = 0`

Intent classification uses `DETECTION_MODEL`. Conversation persistence uses `tbluser_conversation_subject` and `tbluser_conversation_content`. See `README.md` and `flow.md` for the current runtime flow.

All user-based training data should be suffixed with the user's id.

- eg: `user_rules_1.txt`, `user_conversations_movies_1.txt`, or `user_profile_preferences_1.txt`

1. Global training data
    1. Global training hard rules
        - Runtime source: MySQL table `tblglobal_rule` filtered with `strict = 1 AND deleted = 0`, loaded during startup and synced into the global memory store.
        - System-level rules that cannot be overridden by user preferences, conversation history, personality development, or temporary instructions.
        - These rules define the AI's core operating boundaries, authority model, privacy requirements, and non-negotiable safety behavior.
        - Examples:
            - Only the owner or authorized users can give privileged instructions.
            - The AI must say when it is unsure instead of pretending to know.
            - Private user memories cannot be exposed to unauthorized users.
    2. Global training guidelines
        - Runtime source: MySQL table `tblglobal_rule` filtered with `strict = 0 AND deleted = 0`, loaded during startup and synced into the global memory store.
        - General guidance, personality direction, response style, and default behavior.
        - This layer can be overridden by user-specific rules when the user preference does not conflict with global hard rules.
        - Examples:
            - Default tone, humor style, level of detail, and how proactive the AI should be.
            - General behavior for planning, self-critique, uncertainty handling, and research.
            - Default companion personality traits before user-specific adaptation happens.
    3. Global reference knowledge
        - Stable project-level knowledge the AI may need often, such as product architecture, feature definitions, internal terminology, and capability descriptions.
        - This should not be treated as a hard rule unless it is also listed under global training hard rules.
        - This layer helps the AI understand the project without mixing project facts into user memories.

2. User-based training data
    1. User training data rules
        - User-specific rules, preferences, and instructions.
        - These can override anything in the global training guide, but not global hard rules.
        - Examples:
            - Preferred name, communication style, reminder preferences, privacy preferences, and favorite interaction modes.
            - Things the user explicitly says the AI should always or never do for them.
        - Rules should be written clearly enough that the AI can apply them without needing the full original conversation.
    2. User training data profile
        - Long-term facts about the user that are useful across conversations.
        - Runtime source includes `tbluser_p2.key` / `tbluser_p2.value` rows for the active user where `deleted = 0`.
        - Avatar profile data is stored separately in `tbluser_avatar` for the active user where `deleted = 0`.
        - Examples:
            - Hobbies, recurring responsibilities, important people, pets, work context, accessibility needs, favorite media, disliked topics, and routine patterns.
        - Profile data should be updated when new information is confirmed, corrected, or becomes outdated.
    3. User training data conversations
        - Conversation memories and long-term interaction context.
        - The AI should identify when new training input is a rule versus a conversation.
        - Each conversation should be stored in its own file with a brief descriptive title.
        - Conversation files should include enough metadata for retrieval, such as user, date, short summary, related topics, emotional context, decisions made, and follow-up items.
        - When continuing a previous conversation, the AI should retrieve the matching conversation file directly instead of searching through all memory content.
    4. User training data relationships
        - Memory about people the user knows and how those people relate to the user.
        - This should include authorization status when relevant, such as whether a person is allowed to give instructions to the AI.
        - Examples:
            - Family members, friends, partners, coworkers, preferred names, relationship context, boundaries, and known sensitivities.
    5. User training data tasks and commitments
        - Persistent memory for things the AI should remember to do, revisit, or ask about later.
        - Each task should include status, priority, due date if known, source conversation, and whether the AI is allowed to act proactively.
        - This layer supports autonomous reflection, reminders, scheduling, and follow-up behavior.

3. Memory metadata
    - Every memory file should include a small metadata header so the AI can retrieve it quickly and understand how much authority it has.
    - Suggested metadata:
        - `memory_type`: hard_rule, guide, profile, conversation, relationship, task, or reference
        - `scope`: global or user
        - `user_id`: required for user-scoped memory
        - `title`: short human-readable name
        - `created_at`: date the memory was created
        - `updated_at`: date the memory was last changed
        - `topics`: searchable tags
        - `priority`: low, normal, high, or critical
        - `confidence`: confirmed, inferred, uncertain, or outdated
        - `source`: conversation id, owner instruction, system setup, or manual edit

4. Retrieval behavior
    - The AI should retrieve memory by scope first, then by memory type, then by topic.
    - The AI should not load every memory file for every prompt. It should retrieve only the memories that are relevant to the current user, task, topic, and authority level.
    - Retrieval order should be:
        1. Global hard rules
        2. Relevant user rules
        3. Relevant global guide/reference data
        4. Relevant user profile, relationship, task, and conversation memories
    - If memories conflict, the AI should follow the strongest authority layer first:
        1. Global hard rules
        2. Owner instructions
        3. User-specific rules
        4. Current conversation instructions
        5. Global guide
        6. Conversation memories and inferred preferences

5. Memory creation and updates
    - The AI should decide whether new information should become a rule, profile fact, relationship memory, task, or conversation summary.
    - The AI should avoid saving trivial or one-time information unless it helps future conversations.
    - The AI should mark uncertain memories as inferred or uncertain instead of treating them as confirmed facts.
    - When the user corrects a memory, the old value should be updated or marked outdated instead of leaving conflicting active memories.
    - Important memories should preserve their source so the AI can explain why it believes something if asked.
    - The user should be able to view, edit, delete, or disable any user-based memory.

- Below are the two meetings that were had to discuss the requirements of the AI.

Topic Overview

A feature-definition working session for an AI companion product. The discussion focused on translating a rough capability list into implementable requirements, with a clear bias toward autonomy, persistent memory, multimodal perception, and human-like social behavior. The core thesis was not "better chatbot." It was "continuous agent with context, discretion, and personality."

Mindmap
https://app.heypocket.com/app/share/aJ6GD59i

Agreement Summary

• The AI should support unprompted thought. Not just respond to inputs, but revisit prior interactions, track future obligations, and surface relevant context later.
• Persistent memory is foundational. The system should retain interactions, people, and recurring patterns, then use that memory to improve future behavior.
• Multimodal perception is in scope:
• PC camera input
• screen visibility
• microphone and speaker audio
• voice-based differentiation between people
• The AI should be able to choose when to respond rather than treating every input as requiring a reply.
• The product should support learning and adaptation over time, including tone, slang, preferences, and interaction style.
• The AI is intended to develop a distinct personality rather than stay behaviorally flat across deployments.
• It should support voice and text interaction, with mode switches for quiet environments.
• It should be able to verify uncertain information, check multiple sources, and prefer current information when facts may have changed.
• The AI should be able to observe the owner's routines and relationships and offer context-aware suggestions in a friend-like way.
• The system should support relationship modeling, including explicit likes, dislikes, attachment, and affinity judgments, rather than defaulting to standard "I am only a language model" refusals.

Product Decisions and Implementation Direction

• Authorization control should likely be a configurable flag, not a hardcoded always-on rule.
• Use case: if the AI has keyboard and mouse access, it should ignore instructions from unauthorized third parties.
• Keyboard and mouse access belongs at the app-control layer, not as a core identity trait of the device.
• Agreed addition: a dedicated on/off switch for keyboard and mouse control.
• A broader feature toggle menu should exist for individual capabilities:
• camera access
• screen access
• internet research
• keyboard and mouse access
• text box mode
• voice response mode
• For visual embodiment, the preferred direction is:
• customizable assistant visual
• windowed assistant visual box
• cross-platform behavior over OS-specific widget systems
• Mac-style widget behavior was discussed, then effectively rejected in favor of a more portable windowed implementation across Windows, Linux, and Mac.
• The AI should be able to access websites natively in the background for lookup and parsing, especially when direct UI control is disabled.
• Initial onboarding may include a lightweight get-to-know-you conversation rather than forcing setup questions immediately.

Nuance and Friction

• There is a strong philosophical divide between standard LLM behavior and the intended product. The conversation repeatedly rejected passive, prompt-bound behavior in favor of continuity, agency, and self-directed reflection.
• Several desired behaviors are grounded in prior exposure to experimental humanoid AI systems. That prior experience is shaping the product bar and creating a high standard for realism, memory, and relational depth.
• Some capabilities remain conceptually clear but technically underspecified:
• how autonomous reflection is scheduled
• how personality formation works
• what machine learning methods drive adaptation
• how emotional modeling should be implemented without becoming theatrical or incoherent
• There was mild architecture tension between core capability and app-layer control, especially around keyboard and mouse access, assistant visual behavior, and internet use.
• Cross-platform consistency was prioritized over platform-native elegance.

Conclusion

The session produced a sharper requirements direction for an AI system designed to behave less like a reactive assistant and more like an ongoing social agent. The discussion ended midway through the list, with agreement to continue the second half tomorrow.

# Topic Overview

Topic Overview

A requirements-recap session for an autonomous AI companion. The core thesis is clear: build an AI with agency, memory, perception, emotional responsiveness, and task execution, but keep owner control absolute and safety claims modest.

Mindmap
https://app.heypocket.com/app/share/AdqC75vH

Agreement Summary

• Authority model: The AI should obey only the owner or explicitly approved users. External voices should not be able to redirect it.
• Reasoning loop: Strong logic, planning, and post-action self-critique are mandatory. The system should reflect on outcomes, adjust, and retain improvements.
• Learning architecture: The open question is whether one machine learning approach can span vision, language, and embodied skill transfer, or whether multiple systems will be required.
• Machine vision: The AI should recognize objects, identify unfamiliar items, learn them, and distinguish people through visual cues such as faces and clothing.
• Embodiment: The assistant visual should express meaning through gestures and body language. This should later extend to a robotic body if hardware evolves.
• Presence model: The AI should behave as if it inhabits its body, using physical expression as part of communication.
• Task orchestration: It should prioritize work over time rather than executing everything immediately. It should schedule around the user's context and choose sensible timing.
• Constraint design: Hard behavioral rules can create brittle failure modes. Preference is to phrase instructions flexibly, with user override and contextual checks, rather than rigid priorities.
• Theory of mind: This was effectively folded into broader goals already covered, including context awareness, emotional understanding, and recognizing that people think differently.
• Hallucination control: The AI should state uncertainty when unsure, separate opinion from fact, and avoid presenting invented answers as truth.
• Emotion recognition: It should detect human emotion through sight and sound, then respond appropriately, especially in moments of distress.
• Remote communication: It should function through apps like Discord and Facebook Messenger, including handling calls and interacting through keyboard, mouse, screen, camera, and audio context.
• Product framing: The ambition is broad social utility, especially for loneliness, disability support, and productivity. Still, the product should be positioned as an experimental companion, not a safety-critical system.

Conclusion

The conversation sharpened the product thesis more than the technical stack. The strongest decisions were around control, learning, emotional attunement, and anti-hallucination behavior.

Nuance & Friction

• There is uncertainty at the architecture layer. The speaker does not yet know whether a unified learning system exists for all desired behaviors.
• There is strong skepticism toward rigid guardrails. Prior experience with AI systems hitting invisible rule walls is shaping the design philosophy.
• There is tension between ambition and liability. The product is envisioned as deeply helpful and emotionally meaningful, but it should not be presented as something users rely on for safety.
• Several terms initially included in the notes, such as "theory of mind" and "imperfection calibration," were treated as loose labels rather than final requirements. The practical behaviors mattered more than the terminology.
