import { ArrowLeft, Telescope, Github, Linkedin } from "lucide-react";
import { useNavigate } from "react-router-dom";

const About = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background starfield">
      <div className="mx-auto max-w-xl px-6 py-12">
        <button
          onClick={() => navigate("/")}
          className="mb-8 flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft size={16} /> Back to chat
        </button>

        <div className="flex items-center gap-3 mb-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent/60 ring-1 ring-primary/10">
            <Telescope size={20} className="text-primary/70" />
          </div>
          <h1 className="text-2xl font-semibold text-foreground">PointSource</h1>
        </div>

        <section className="space-y-4 text-sm leading-relaxed text-foreground/80">
          <p>
            PointSource is an AI-powered assistant specializing in astrophysics.
            It helps researchers, students, and space enthusiasts explore astrophysics and astronomy topics,
            with cited, verifiable answers from scientific literature.
          </p>

          <h2 className="text-xs uppercase tracking-wide text-muted-foreground pt-4">About version alpha</h2>
          <p className="text-foreground/70">
            PointSource is currently in alpha. The papers available to the system are a curated subset of
            arXiv rather than the full corpus: specifically, papers in the <code>astro-ph</code> category from
            the{" "}
            <a
              href="https://ar5iv.labs.arxiv.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:opacity-70 transition-opacity"
            >
              ar5iv
            </a>{" "}
            HTML project, filtered to documents without parsing errors to prioritize answer quality and citation
            reliability. The current source dataset is{" "}
            <a
              href="https://huggingface.co/datasets/marin-community/ar5iv-no-problem-markdown"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:opacity-70 transition-opacity"
            >
              marin-community/ar5iv-no-problem-markdown
            </a>.
          </p>

          <h2 className="text-xs uppercase tracking-wide text-muted-foreground pt-4">Tech stack</h2>
          <ul className="list-disc pl-5 space-y-1 text-foreground/70">
            <li>React + TypeScript frontend built with Vite and Tailwind CSS</li>
            <li>FastAPI backend with Pydantic AI and LiteLLM model gateway</li>
            <li>PostgreSQL + pgvector for data and vector search, with Redis for runtime services</li>
            <li>RAG pipeline with scientific-document retrieval and streamed responses</li>
          </ul>

          <h2 className="text-xs uppercase tracking-wide text-muted-foreground pt-4">Contact</h2>
          <div className="flex items-center gap-4 text-foreground/70">
            <a
              href="https://github.com/Alexperiments/point-source"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-foreground transition-colors"
            >
              <Github size={15} /> GitHub
            </a>
            <a
              href="https://www.linkedin.com/in/alessandro--diana/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-foreground transition-colors"
            >
              <Linkedin size={15} /> LinkedIn
            </a>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;
