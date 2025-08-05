Problem Analysis
Objective
Create a browser-based JavaScript/TypeScript application to help students prepare for exams by:
	• Ingesting local study materials (notes, readings, lecture transcripts).
	• Automatically generating multiple-choice questions (MCQs) covering key concepts.
	• Tracking user performance on each topic.
	• Visualizing strengths and weaknesses via a heat map.
	• Recommending targeted reading from the provided material to shore up weak areas.
Key Stakeholders
	• Students (primary users)
	• Educators / Content Providers
High-Level Requirements
	1. Content Ingestion
		○ Allow users to select folders/files from local storage.
		○ Support common formats: TXT, PDF, MD, DOCX.
	2. Content Processing
		○ Extract text, split into logical units (sections, paragraphs).
		○ Identify key concepts and facts.
	3. MCQ Generation
		○ Use an AI/NLP service (on-device or cloud) to: 
			§ Generate question stem.
			§ Create one correct answer and multiple distractors.
		○ Allow manual review/editing of generated questions.
	4. Quiz Engine
		○ Present MCQs in randomized order.
		○ Record user responses, timing, and confidence (optional).
	5. Performance Visualization
		○ Aggregate results by topic or section.
		○ Display a heat map grid showing high/medium/low mastery.
	6. Recommendation Engine
		○ For low-scoring topics, identify relevant source passages.
		○ Present “Next Reading” suggestions with direct links to content.
	7. Persistence & Offline Support
		○ Store quizzes, results, and indexed content in browser storage (IndexedDB).
		○ Work fully offline once the application is loaded.
	8. UX/UI
		○ Clean, responsive interface (desktop/tablet).
		○ Accessible (WCAG AA).

Technical Challenges & Considerations
	• Client-only File Access
Leverage the File System Access API for folder and file reads. Fallback to file input for unsupported browsers.
	• Text Extraction
Use libraries like pdf.js and mammoth.js to parse PDFs and DOCX respectively.
	• AI/NLP MCQ Generation
Decide between:
		○ Calling a cloud API (e.g., OpenAI)
		○ Bundling a lightweight on-device model (e.g., Bloom in WebAssembly)
	• Topic Modeling
Tag questions and source passages by topic using keyword extraction or NLP libraries (e.g., compromise, wink-nlp).
	• Heat Map Visualization
Employ D3.js or Chart.js to render an interactive matrix where rows are topics and columns are quiz sessions.
	• Data Storage
IndexedDB via idb library for structured storage of content, quizzes, and results.
	• State Management
Use Redux or Zustand to manage app state (content index, quiz state, performance metrics).
![Uploading image.png…]()
