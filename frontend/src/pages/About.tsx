import { ArrowLeft, Telescope, Github, Linkedin, ExternalLink } from "lucide-react";
import { useNavigate } from "react-router-dom";

const About = () => {
  const navigate = useNavigate();
  const repoUrl = "https://github.com/Alexperiments/point-source";
  const linkedInUrl = "https://www.linkedin.com/in/alessandro--diana/";
  const paypalUrl = "https://paypal.me/wiGh63";

  const supportButtonClass =
    "inline-flex items-center gap-1.5 rounded-lg border border-input bg-background px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent";

  const faqs = [
    {
      question: "What kind of questions can I ask to point-source?",
      answer: (
        <p>
          Ask about astrophysics and astronomy papers, concepts, observations, methods, and terminology.
          PointSource works best for literature-grounded requests such as paper summaries, comparisons between
          results, explanations of technical ideas, and answers that need citations back to the source papers.
        </p>
      ),
    },
    {
      question: "How is point-source different from ChatGPT, and similar models?",
      answer: (
        <p>
          PointSource is narrower in scope and optimized for astrophysics research. Instead of acting as a
          general-purpose assistant, it focuses on answers grounded in a curated scientific corpus and surfaces
          citations so you can inspect the evidence. It is similar to general chat models in that you can ask
          follow-up questions conversationally and get plain-language explanations.
        </p>
      ),
    },
    {
      question: "What&apos;s a RAG?",
      answer: (
        <p>
          RAG stands for retrieval-augmented generation. In practice, that means the system first retrieves
          relevant passages from its document collection and then uses those passages to generate an answer,
          rather than relying only on the model&apos;s internal memory.
        </p>
      ),
    },
    {
      question: "Why is there a daily limit to the questions I can ask?",
      answer: (
        <p>
          The daily limit helps keep the service stable and affordable while PointSource is still in alpha. It
          reduces abuse, protects shared compute resources, and makes it easier to iterate on answer quality
          before opening usage more broadly.
        </p>
      ),
    },
    {
      question: "How can I contribute?",
      answer: (
        <p>
          Code contributions, bug reports, docs fixes, and product feedback are all useful. If you want to
          contribute code, start with the{" "}
          <a
            href={repoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:opacity-70 transition-opacity"
          >
            GitHub repository
          </a>{" "}
          and open an issue or pull request. If you want to support the project financially, you can use PayPal
          below.
        </p>
      ),
      actions: [
        {
          href: repoUrl,
          label: "Contribute on GitHub",
          icon: <Github size={15} />,
        },
        {
          href: paypalUrl,
          label: "Support via PayPal",
          icon: <ExternalLink size={15} />,
        },
      ],
    },
  ];

  return (
    <div className="min-h-screen bg-background starfield">
      <div className="mx-auto max-w-2xl px-6 py-12">
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
            arXiv rather than the full corpus: specifically, 81756 papers in the <code>astro-ph</code> category from
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

          <h2 className="text-xs uppercase tracking-wide text-muted-foreground pt-4">FAQ</h2>
          <div className="space-y-3">
            {faqs.map((faq) => (
              <article
                key={faq.question}
                className="rounded-2xl border border-border/80 bg-card/70 px-4 py-4 shadow-sm"
              >
                <h3 className="text-sm font-semibold text-foreground">{faq.question}</h3>
                <div className="mt-2 text-foreground/70">{faq.answer}</div>
                {faq.actions ? (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {faq.actions.map((action) => (
                      <a
                        key={action.label}
                        href={action.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={supportButtonClass}
                      >
                        {action.icon}
                        {action.label}
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
          </div>

          <h2 className="text-xs uppercase tracking-wide text-muted-foreground pt-4">Contact</h2>
          <p className="text-foreground/70">
            If you want to follow the project, contribute code, or support it directly, use the links below.
          </p>
          <div className="flex flex-wrap items-center gap-2 text-foreground/70">
            <a
              href={repoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={supportButtonClass}
            >
              <Github size={15} /> GitHub
            </a>
            <a
              href={linkedInUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={supportButtonClass}
            >
              <Linkedin size={15} /> LinkedIn
            </a>
            <a
              href={paypalUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={supportButtonClass}
            >
              <ExternalLink size={15} /> PayPal
            </a>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;
