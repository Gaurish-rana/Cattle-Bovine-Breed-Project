import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Camera, Upload, Loader2, Download, Share2, AlertTriangle, 
  ChevronDown, ChevronUp, RefreshCw, X, CheckCircle2 
} from 'lucide-react';
import { API_BASE_URL, PredictionResult, RankedProbability, APIResponse } from '@/lib/mockData';
import { useToast } from '@/hooks/use-toast';

const Identify = () => {
  const [selectedImage, setSelectedImage] = useState<File | Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [allProbabilities, setAllProbabilities] = useState<RankedProbability[]>([]);
  const [apiMessage, setApiMessage] = useState<string>('');
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [architecture, setArchitecture] = useState<string>('');
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const [disclaimer, setDisclaimer] = useState<string>('');
  const [showAllClasses, setShowAllClasses] = useState<boolean>(false);

  // Webcam State
  const [isWebcamOpen, setIsWebcamOpen] = useState(false);
  const [cameraError, setCameraError] = useState<string>('');
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  // Clean up preview URLs and webcam stream on unmount
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
      stopWebcam();
    };
  }, [previewUrl]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 15 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image under 15MB",
          variant: "destructive"
        });
        return;
      }
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
      setAllProbabilities([]);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 15 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please select an image under 15MB",
          variant: "destructive"
        });
        return;
      }
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
      setAllProbabilities([]);
    }
  };

  // Webcam Controls
  const startWebcam = async () => {
    setCameraError('');
    setIsWebcamOpen(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setCameraError('Unable to access camera. Please verify device permissions.');
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsWebcamOpen(false);
  };

  const captureWebcamFrame = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          stopWebcam();
          setSelectedImage(blob);
          setPreviewUrl(URL.createObjectURL(blob));
          setResults([]);
          setAllProbabilities([]);
        }
      }, 'image/jpeg', 0.92);
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setResults([]);
    setAllProbabilities([]);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage, 'upload.jpg');

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data: APIResponse = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server responded with status ${response.status}`);
      }

      if (data.success && data.predictions) {
        setResults(data.predictions);
        setAllProbabilities(data.all_probabilities || []);
        setApiMessage(data.message || 'Real AI Model Active');
        setModelLoaded(data.model_loaded || true);
        setArchitecture(data.architecture || 'ConvNeXt-Tiny');
        setInferenceTime(data.inference_time_ms || null);
        setDisclaimer(data.disclaimer || '');

        const topBreed = data.predictions[0];
        toast({
          title: `✅ Identified: ${topBreed.breed_name} (${topBreed.species})`,
          description: `Confidence: ${Math.round(topBreed.confidence * 100)}% via ${data.architecture || 'ConvNeXt-Tiny'}`
        });
      } else {
        throw new Error(data.error || 'Prediction response missing valid classification data.');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Backend connection failed. Please ensure the Flask API is running.';
      
      toast({
        title: "Identification Failed",
        description: errorMessage,
        variant: "destructive"
      });

      setResults([{
        breed_name: "Connection Error",
        species: "Cattle",
        confidence: 0,
        breed_id: "error",
        characteristics: {
          origin: "N/A",
          primary_use: "Verify Flask server is running at http://localhost:5000",
          average_weight: "N/A",
          distinctive_features: errorMessage
        }
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.70) return "text-green-600";
    if (confidence >= 0.45) return "text-yellow-600";
    return "text-red-500";
  };

  return (
    <div className="min-h-screen gradient-subtle">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-foreground mb-4">
              Identify Your Cattle & Buffalo
            </h1>
            <p className="text-lg text-muted-foreground mb-4">
              Upload an image or scan live to identify among 17 indigenous breeds with AI precision
            </p>
            {apiMessage && (
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm bg-green-100 text-green-800 border border-green-200">
                <span className="w-2 h-2 rounded-full bg-green-600 animate-pulse"></span>
                <span className="font-medium">{apiMessage}</span>
                {inferenceTime && (
                  <span className="text-xs text-green-700 ml-1">({inferenceTime}ms)</span>
                )}
              </div>
            )}
          </div>

          {/* Webcam Modal */}
          {isWebcamOpen && (
            <Card className="mb-8 border-2 border-primary/40 shadow-card">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Camera className="w-5 h-5 text-primary" />
                  Live Camera Scanner
                </CardTitle>
                <Button variant="ghost" size="sm" onClick={stopWebcam}>
                  <X className="w-4 h-4" />
                </Button>
              </CardHeader>
              <CardContent className="space-y-4">
                {cameraError ? (
                  <div className="p-4 bg-red-50 text-red-700 rounded-lg text-sm">
                    {cameraError}
                  </div>
                ) : (
                  <div className="relative aspect-video max-w-lg mx-auto bg-black rounded-lg overflow-hidden">
                    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                  </div>
                )}
                <div className="flex justify-center gap-3">
                  <Button variant="outline" onClick={stopWebcam}>
                    Cancel
                  </Button>
                  <Button variant="farm" onClick={captureWebcamFrame} disabled={Boolean(cameraError)}>
                    <Camera className="w-4 h-4 mr-2" />
                    Capture & Analyze
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Upload Interface */}
          <Card className="mb-8 shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload Image
                </span>
                <Button variant="outline" size="sm" onClick={startWebcam} className="text-xs flex items-center gap-1.5">
                  <Camera className="w-4 h-4 text-primary" />
                  Use Webcam
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!selectedImage ? (
                <div
                  className="border-2 border-dashed border-border rounded-xl p-12 text-center hover:border-primary transition-smooth cursor-pointer"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Camera className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium text-foreground mb-2">
                    Drag and drop your cattle image here
                  </p>
                  <p className="text-muted-foreground mb-4">
                    or click to browse local files
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Supports JPG, PNG, WebP, BMP (Max 15MB)
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <img
                      src={previewUrl}
                      alt="Selected cattle"
                      className="w-full max-w-md mx-auto rounded-lg shadow-card object-contain max-h-96"
                    />
                  </div>
                  <div className="flex justify-center gap-3">
                    <Button 
                      variant="outline" 
                      onClick={() => {
                        setSelectedImage(null);
                        setPreviewUrl('');
                        setResults([]);
                        setAllProbabilities([]);
                      }}
                      disabled={isProcessing}
                    >
                      Remove Image
                    </Button>
                    <Button 
                      variant="farm"
                      onClick={processImage}
                      disabled={isProcessing}
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin mr-2" />
                          Classifying with {architecture || 'ConvNeXt-Tiny'}...
                        </>
                      ) : (
                        'Identify Breed'
                      )}
                    </Button>
                  </div>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </CardContent>
          </Card>

          {/* Results Section */}
          {results.length > 0 && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">
                    Classification Results
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    Analyzed by {architecture} in {inferenceTime || '—'} ms
                  </p>
                </div>
              </div>
              
              {/* Primary Prediction Cards */}
              <div className="grid gap-4">
                {results.map((result, index) => (
                  <Card key={`${result.breed_id}-${index}`} className="shadow-card hover:shadow-elegant transition-smooth border-l-4 border-l-primary">
                    <CardContent className="p-6">
                      {result.breed_id === 'error' ? (
                        <div className="flex items-start gap-4">
                          <AlertTriangle className="w-8 h-8 text-red-500 mt-1 shrink-0" />
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold text-red-600 mb-2">
                              {result.breed_name}
                            </h3>
                            <div className="space-y-2 text-sm">
                              <p className="font-medium text-foreground">{result.characteristics.primary_use}</p>
                              <p className="text-muted-foreground">{result.characteristics.distinctive_features}</p>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="flex items-start justify-between mb-4">
                            <div>
                              <div className="flex items-center gap-2">
                                <h3 className="text-2xl font-bold text-foreground">
                                  {result.breed_name}
                                </h3>
                                <Badge variant={result.species === 'Cattle' ? 'default' : 'secondary'} className="text-xs">
                                  {result.species}
                                </Badge>
                                {result.characteristics?.accession_number && (
                                  <span className="text-xs text-muted-foreground font-mono hidden sm:inline">
                                    [{result.characteristics.accession_number}]
                                  </span>
                                )}
                              </div>
                              <Badge variant="outline" className="mt-1.5 text-xs font-medium">
                                {index === 0 ? '🏆 Primary Prediction' : `Alternate Match #${index + 1}`}
                              </Badge>
                            </div>
                            <div className="text-right">
                              <div className={`text-3xl font-extrabold ${getConfidenceColor(result.confidence)}`}>
                                {Math.round(result.confidence * 100)}%
                              </div>
                              <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Confidence</p>
                            </div>
                          </div>
                          
                          <Progress 
                            value={result.confidence * 100} 
                            className="mb-4 h-2.5"
                          />
                          
                          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-sm bg-muted/30 p-3.5 rounded-lg">
                            <div>
                              <p className="font-semibold text-foreground text-xs uppercase tracking-wide">Origin / State</p>
                              <p className="text-muted-foreground mt-0.5">{result.characteristics.origin}</p>
                            </div>
                            <div>
                              <p className="font-semibold text-foreground text-xs uppercase tracking-wide">Primary Utility</p>
                              <p className="text-muted-foreground mt-0.5">{result.characteristics.primary_use}</p>
                            </div>
                            <div>
                              <p className="font-semibold text-foreground text-xs uppercase tracking-wide">Typical Weight</p>
                              <p className="text-muted-foreground mt-0.5">{result.characteristics.average_weight}</p>
                            </div>
                            <div>
                              <p className="font-semibold text-foreground text-xs uppercase tracking-wide">Breed Hallmarks</p>
                              <p className="text-muted-foreground mt-0.5">{result.characteristics.distinctive_features}</p>
                            </div>
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Complete 17-Class Breakdown Accordion */}
              {allProbabilities.length > 0 && (
                <Card className="shadow-card mt-6">
                  <CardHeader 
                    className="cursor-pointer py-4 flex flex-row items-center justify-between"
                    onClick={() => setShowAllClasses(!showAllClasses)}
                  >
                    <CardTitle className="text-lg font-semibold flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-primary" />
                      Full 17-Class Probability Distribution
                    </CardTitle>
                    <Button variant="ghost" size="sm">
                      {showAllClasses ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </Button>
                  </CardHeader>

                  {showAllClasses && (
                    <CardContent className="pt-0 space-y-3">
                      <p className="text-xs text-muted-foreground mb-3">
                        Ranked posterior softmax probabilities over all 17 indigenous cattle and buffalo breeds:
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {allProbabilities.map((item) => (
                          <div key={item.breed_id} className="p-2.5 rounded-lg border border-border/60 bg-background/50 flex flex-col justify-between">
                            <div className="flex items-center justify-between mb-1.5 text-xs">
                              <span className="font-medium text-foreground">
                                #{item.rank} {item.breed_name} <span className="text-muted-foreground">({item.species})</span>
                              </span>
                              <span className={`font-bold ${item.confidence >= 0.1 ? 'text-primary' : 'text-muted-foreground'}`}>
                                {item.confidence_percentage}
                              </span>
                            </div>
                            <Progress value={item.confidence * 100} className="h-1.5" />
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  )}
                </Card>
              )}

              {/* Disclaimer */}
              {disclaimer && (
                <p className="text-xs text-center text-muted-foreground italic px-4">
                  {disclaimer}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Identify;
