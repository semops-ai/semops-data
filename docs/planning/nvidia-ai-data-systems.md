https://podcasts.apple.com/us/podcast/nvidia-ai-podcast/id1186480811?i=1000737

Real data / MLOps
AI needs access to data
“AI ready data”
RAG, “Data Flywheel”, Agents
Challenging - unstructured , PowerPoint, etc

Pipeline to find, gather, make AI ready “lift and sift”

https://www.nvidia.com/en-us/data-center/ai-data-platform/

https://blogs.nvidia.com/blog/ai-data-platform-gpu-accelerated-storage/?ncid=no-ncid 

AI agents have the potential to become indispensable tools for automating complex tasks. But bringing agents to production remains challenging.

According to Gartner, “about 40% of AI prototypes make it into production, and participants reported data availability and quality as a top barrier to AI adoption.1”

Just like human workers, AI agents need secure, relevant, accurate and recent data to deliver business value — what the industry is now calling “AI-ready data.”

Making enterprise data AI-ready presents unique challenges. Gartner estimates, “Unstructured data such as documents and multimedia files accounts for 70% to 90% of organizational data, and poses unique governance challenges due to its volume, variety and lack of coherent structure.2” Unstructured data sources include email, PDFs, videos, audio clips and presentations.

An emerging class of GPU-accelerated data and storage infrastructure — the AI data platform — transforms unstructured data into AI-ready data quickly and securely.

What Is AI-Ready Data?
AI-ready data can be consumed by AI training, fine-tuning and retrieval-augmented generation pipelines without additional preparation.

Making unstructured data AI-ready involves:

Collecting and curating data from diverse sources
Applying metadata for data management and governance
Dividing the source documents into semantically relevant chunks
Embedding the chunks into vectors for efficient storage, search and retrieval
Enterprises cannot unlock the full value of their AI investments until their unstructured data is AI-ready.

Why Making Data AI-Ready Is Difficult
Making unstructured data AI-ready remains a substantial challenge for most enterprises due to:

Data complexity: A typical enterprise has hundreds of diverse data sources in dozens of formats and modalities — including video, audio, text and images. This data lives in different storage silos.
Data velocity: The volume of business data is exploding. Predictions show global stored data will double over the next four years. And the rate of data change is increasing as enterprises adopt real-time streaming sensors such as camera feeds.
Data sprawl and data drift: Frequent data copying and transformation introduces cost and security risks. Over time, the content or permissions of AI representations — such as text chunks and embeddings — diverge from source-of-truth documents. Plus, as the number of chatbots and agents proliferates, the security risk of data increases.
Together, these factors force enterprise data scientists to spend the majority of their time locating, cleaning and organizing data — leaving less time for identifying valuable insights.

The AI Data Platform — a New Class of Enterprise Data and Storage Infrastructure
AI data platforms are an emerging class of GPU-accelerated data and storage infrastructure that makes enterprise data AI-ready.

By embedding GPU acceleration directly into the data path, AI data platforms transform data for AI pipelines as a background operation invisible to the user.

The data is prepared in place, minimizing unnecessary copies and associated security risks.

By integrating data preparation as a core capability of storage infrastructure, AI data platforms ensure that the accuracy and security of the data is maintained. Any modifications to the sources of truth documents — including edits or permission changes — are instantly conveyed to their associated vector embeddings.

Key benefits of AI data platforms include:

Faster time to value: Enterprises don’t need to design, build and optimize AI data pipelines from the ground up. AI data platforms deliver an integrated, state-of-the-art AI data pipeline out of the box.
Reduced data drift: By continuously ingesting, embedding and indexing enterprise data in near real time, AI data platforms reduce time to insight and minimize data drift.
Improved data security: Because source-of-truth documents are stored together in AI data platforms, any changes to their contents or permissions are instantly propagated to the AI applications that use them.
Simplified data governance: Preparing data in place reduces the proliferation of shadow copies that undermine access control, traceability and compliance.
Improved GPU utilization: In an AI data platform, GPU capacity is sized for the amount, type and change velocity of the data under management. GPU capacity scales with the data, ensuring GPUs are not over- or under-provisioned for data preparation tasks.
The NVIDIA AI Data Platform
AI is changing every industry — and AI data platforms are the natural evolution of enterprise storage for the generative AI era, changing from passive containers to active engines delivering business value.

By integrating GPU acceleration into the data path, AI data platforms enable enterprises to activate their AI agents with AI-ready data quickly and securely.

The NVIDIA AI Data Platform reference design brings together NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, NVIDIA BlueField-3 DPUs and integrated AI data processing pipelines based on NVIDIA Blueprints.

The NVIDIA AI Data Platform design has been adopted by leading AI infrastructure and storage providers including Cisco, Cloudian, DDN, Dell Technologies, Hitachi Vantara, HPE, IBM, NetApp, Pure Storage, VAST Data and WEKA — each extending the design with their own unique differentiation and innovation.