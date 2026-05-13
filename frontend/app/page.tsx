"use client";

import { ChangeEvent, FormEvent, useMemo, useRef, useState } from "react";
import Image from "next/image";
import { Search, ImageIcon, Mic, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

type SearchResult = {
  image_uri: string;
  name: string;
  brand: string;
  price: number;
  color: string;
  description: string;
  attributes: string;
};

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  0: {
    transcript: string;
  };
};

type SpeechRecognitionEventLike = Event & {
  results: ArrayLike<SpeechRecognitionResultLike>;
};

type SpeechRecognitionLike = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function toImageUrl(imageUri: string): string {
  const normalized = imageUri.replaceAll("\\", "/").replace(/^\.?\//, "");
  if (normalized.startsWith("http://") || normalized.startsWith("https://")) return normalized;
  return `${API_BASE}/${normalized}`;
}

export default function Home() {
  const [query, setQuery] = useState("kurta");
  const [limit, setLimit] = useState(10);
  const [useHybrid, setUseHybrid] = useState(false);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [languageCode, setLanguageCode] = useState("en-IN");
  const [isListening, setIsListening] = useState(false);
  const [supportsRecognition, setSupportsRecognition] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const transcriptRef = useRef("");

  const [results, setResults] = useState<SearchResult[]>([]);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const hasResults = useMemo(() => results.length > 0, [results]);

  async function onTextSearch(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setTranscript("");
    try {
      const response = await fetch(`${API_BASE}/search/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          limit,
          use_hybrid: useHybrid,
          image_weight: 0.6,
          text_weight: 0.4,
          filters: {},
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? "Text search failed.");
      setResults(payload.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Text search failed.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function onImageSearch(e: FormEvent) {
    e.preventDefault();
    if (!imageFile) {
      setError("Please choose an image file first.");
      return;
    }
    setLoading(true);
    setError("");
    setTranscript("");
    try {
      const form = new FormData();
      form.append("file", imageFile);
      form.append("limit", String(limit));
      form.append("use_hybrid", String(useHybrid));
      form.append("image_weight", "0.6");
      form.append("text_weight", "0.4");

      const response = await fetch(`${API_BASE}/search/image`, { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? "Image search failed.");
      setResults(payload.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image search failed.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function onVoiceSearch(voiceText: string) {
    const normalized = voiceText.trim();
    if (!normalized) return;

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/search/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: normalized,
          limit,
          use_hybrid: useHybrid,
          image_weight: 0.6,
          text_weight: 0.4,
          filters: {},
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? "Voice search failed.");
      setResults(payload.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Voice search failed.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function onImageChange(e: ChangeEvent<HTMLInputElement>) {
    setImageFile(e.target.files?.[0] ?? null);
  }

  function getRecognitionCtor(): SpeechRecognitionCtor | null {
    if (typeof window === "undefined") return null;
    const maybe = window as Window & {
      SpeechRecognition?: SpeechRecognitionCtor;
      webkitSpeechRecognition?: SpeechRecognitionCtor;
    };
    return maybe.SpeechRecognition ?? maybe.webkitSpeechRecognition ?? null;
  }

  function startListening() {
    const Recognition = getRecognitionCtor();
    if (!Recognition) {
      setSupportsRecognition(false);
      setError("Speech recognition is not supported in this browser.");
      return;
    }

    setSupportsRecognition(true);
    setError("");
    setTranscript("");

    const recognition = new Recognition();
    recognition.lang = languageCode;
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onresult = (event) => {
      let combined = "";
      for (let i = 0; i < event.results.length; i += 1) {
        combined += event.results[i][0].transcript;
      }
      const next = combined.trim();
      transcriptRef.current = next;
      setTranscript(next);
    };

    recognition.onerror = () => {
      setError("Could not capture audio. Please allow microphone access and try again.");
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      recognitionRef.current = null;
      if (transcriptRef.current.trim()) {
        void onVoiceSearch(transcriptRef.current);
      }
    };

    recognition.start();
    recognitionRef.current = recognition;
    setIsListening(true);
  }

  function toggleListening() {
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      return;
    }
    startListening();
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto w-full max-w-6xl px-4 py-8 md:py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight md:text-4xl">Fashion Search</h1>
          <p className="mt-2 text-muted-foreground">Find products by text, image, or voice</p>
        </div>

        {/* Search Section */}
        <div className="mb-12 rounded-2xl border border-border bg-white p-8 shadow-sm">
          <Tabs defaultValue="text" className="w-full">
            <TabsList className="mb-6 grid w-full grid-cols-3 bg-muted">
              <TabsTrigger value="text">
                <Search className="mr-2 size-4" /> Text
              </TabsTrigger>
              <TabsTrigger value="image">
                <ImageIcon className="mr-2 size-4" /> Image
              </TabsTrigger>
              <TabsTrigger value="audio">
                <Mic className="mr-2 size-4" /> Voice
              </TabsTrigger>
            </TabsList>

            {/* Text search */}
            <TabsContent value="text" className="mt-0">
              <form onSubmit={onTextSearch} className="space-y-4">
                <Textarea
                  className="min-h-24 resize-none border-border bg-background"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="black kurta with embroidery..."
                />
                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-primary text-white hover:bg-primary/90 disabled:opacity-70"
                >
                  {loading ? <Loader2 className="mr-2 size-4 animate-spin" /> : <Search className="mr-2 size-4" />}
                  Search by Text
                </Button>
              </form>
            </TabsContent>

            {/* Image search */}
            <TabsContent value="image" className="mt-0">
              <form onSubmit={onImageSearch} className="space-y-4">
                <div className="relative rounded-lg border-2 border-dashed border-border p-8 text-center hover:border-primary/40 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={onImageChange}
                    className="absolute inset-0 cursor-pointer opacity-0"
                  />
                  <div className="pointer-events-none">
                    <ImageIcon className="mx-auto mb-2 size-8 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">
                      {imageFile ? imageFile.name : "Drop image or click to browse"}
                    </p>
                  </div>
                </div>
                <Button
                  type="submit"
                  disabled={loading || !imageFile}
                  className="w-full bg-primary text-white hover:bg-primary/90 disabled:opacity-70"
                >
                  {loading ? <Loader2 className="mr-2 size-4 animate-spin" /> : <ImageIcon className="mr-2 size-4" />}
                  Search by Image
                </Button>
              </form>
            </TabsContent>

            {/* Voice search */}
            <TabsContent value="audio" className="mt-0">
              <div className="space-y-4">
                <div className="rounded-lg border-2 border-dashed border-border p-8 text-center">
                  <button
                    type="button"
                    onClick={toggleListening}
                    disabled={loading}
                    className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-primary text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-70"
                    aria-label={isListening ? "Stop voice transcription" : "Start voice transcription"}
                  >
                    {isListening ? <Loader2 className="size-6 animate-spin" /> : <Mic className="size-6" />}
                  </button>
                  <p className="text-sm text-muted-foreground">
                    {isListening ? "Listening... click again to stop" : "Click the mic to start speaking"}
                  </p>
                  {!supportsRecognition ? (
                    <p className="mt-2 text-xs text-muted-foreground">
                      Works in browsers with Web Speech API support (like Chrome/Edge).
                    </p>
                  ) : null}
                </div>
                <label className="block space-y-2">
                  <span className="text-sm font-medium text-foreground">Language</span>
                  <select
                    className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20"
                    value={languageCode}
                    onChange={(e) => setLanguageCode(e.target.value)}
                  >
                    <option value="en-IN">English (en-IN)</option>
                    <option value="hi-IN">Hindi (hi-IN)</option>
                    <option value="mr-IN">Marathi (mr-IN)</option>
                  </select>
                </label>
              </div>
            </TabsContent>
          </Tabs>
        </div>

        {/* Settings */}
        <div className="mb-12 rounded-2xl border border-border bg-white p-8">
          <h3 className="mb-6 text-lg font-semibold">Search Settings</h3>
          <div className="grid gap-6 md:grid-cols-2">
            <label className="space-y-2">
              <span className="block text-sm font-medium text-foreground">Results to show</span>
              <Input
                type="number"
                min={1}
                max={30}
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value || 10))}
                className="border-border"
              />
            </label>
            <label className="flex items-end gap-3 rounded-lg border border-border bg-background px-4 py-3">
              <input
                type="checkbox"
                checked={useHybrid}
                onChange={(e) => setUseHybrid(e.target.checked)}
                className="h-4 w-4 cursor-pointer"
              />
              <span className="text-sm font-medium text-foreground">Enable hybrid reranking</span>
            </label>
          </div>
        </div>

        {/* Error display */}
        {error ? (
          <div className="mb-12 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        ) : null}

        {/* Transcript display */}
        {transcript ? (
          <div className="mb-12 rounded-2xl border border-primary/10 bg-primary/5 p-6">
            <h3 className="mb-3 font-semibold text-foreground">Recognized Transcript</h3>
            <p className="text-sm leading-relaxed text-muted-foreground">{transcript}</p>
          </div>
        ) : null}

        {/* Results */}
        <section>
          <div className="mb-8 flex items-center justify-between border-b border-border pb-6">
            <h2 className="text-2xl font-bold">Results</h2>
            {hasResults && (
              <Badge variant="secondary" className="bg-primary/10 text-primary border-0">
                {results.length} items
              </Badge>
            )}
          </div>

          {hasResults ? (
            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {results.map((item, idx) => (
                <div
                  key={`${item.image_uri}-${idx}`}
                  className="group rounded-lg border border-border bg-white transition-all duration-300 hover:shadow-lg hover:border-primary/30"
                >
                  <div className="relative aspect-square overflow-hidden bg-muted rounded-t-lg">
                    <Image
                      src={toImageUrl(item.image_uri)}
                      alt={item.name || "Result"}
                      fill
                      className="object-cover transition-transform duration-300 group-hover:scale-105"
                      unoptimized
                    />
                  </div>
                  <div className="space-y-3 p-4">
                    <p className="line-clamp-2 text-sm font-semibold leading-tight">{item.name || "Untitled"}</p>
                    <div className="flex items-start justify-between gap-2">
                      <Badge variant="outline" className="max-w-full truncate text-xs">
                        {item.attributes || "-"}
                      </Badge>
                      {item.price > 0 && (
                        <span className="text-xs font-medium text-primary">
                          {new Intl.NumberFormat("en-IN", {
                            style: "currency",
                            currency: "INR",
                            maximumFractionDigits: 0,
                          }).format(item.price)}
                        </span>
                      )}
                    </div>
                    <p className="line-clamp-2 text-xs text-muted-foreground">{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-border bg-background/50 py-16 text-center">
              <p className="text-sm text-muted-foreground">
                Start by searching with text, image, or voice to see results
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
