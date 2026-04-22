import { useState } from 'react';

interface BodyStressMapProps {
  data: { [key: string]: number[][] };
  formScore?: number;
  formErrors?: Array<{
    body_part: string;
    severity: string;
  }>;
}

interface JointStress {
  name: string;
  level: 'low' | 'moderate' | 'high';
  avgTorque: number;
  description: string;
}

export function BodyStressMap({ data, formScore, formErrors }: BodyStressMapProps) {
  const [selectedJoint, setSelectedJoint] = useState<string | null>(null);

  // Calculate stress level based on form score and form errors
  const getJointStress = (jointKey: string): JointStress => {
    const rawData = data[`${jointKey}_torque`];
    
    // Calculate average torque for display (normalize to 0-1 scale)
    let avgTorque = 0;
    if (rawData && Array.isArray(rawData)) {
      const allValues = rawData.flat().filter(v => typeof v === 'number' && v > 0);
      const rawAvg = allValues.length > 0 ? allValues.reduce((a, b) => a + b, 0) / allValues.length : 0;
      // Normalize: torque values typically range 0-150, so divide by 150
      avgTorque = Math.min(1, rawAvg / 150);
    }
    
    // Determine stress level based on form errors for this joint
    let level: 'low' | 'moderate' | 'high' = 'low';
    let description: string;
    
    // Check if there's a form error for this joint
    const jointError = formErrors?.find(
      err => err.body_part.toLowerCase().includes(jointKey.toLowerCase()) ||
             jointKey.toLowerCase().includes(err.body_part.toLowerCase().replace(/s$/, ''))
    );
    
    if (jointError) {
      // Use the severity from form errors
      if (jointError.severity === 'severe') {
        level = 'high';
        description = `High stress on ${jointKey}! Form needs correction. Check the recommendations below.`;
      } else if (jointError.severity === 'moderate') {
        level = 'moderate';
        description = `Moderate stress on ${jointKey}. Minor adjustments needed.`;
      } else {
        level = 'low';
        description = `Low stress on ${jointKey}. Minor improvement possible.`;
      }
    } else if (formScore !== undefined) {
      // Use overall form score if no specific error for this joint
      if (formScore >= 80) {
        level = 'low';
        description = `${jointKey.charAt(0).toUpperCase() + jointKey.slice(1)} form is good! Keep it up.`;
      } else if (formScore >= 60) {
        level = 'moderate';
        description = `${jointKey.charAt(0).toUpperCase() + jointKey.slice(1)} could use some improvement.`;
      } else {
        level = 'high';
        description = `${jointKey.charAt(0).toUpperCase() + jointKey.slice(1)} needs attention. Review form tips.`;
      }
    } else {
      description = 'No specific data available for this joint.';
    }
    
    return { name: jointKey, level, avgTorque, description };
  };

  const joints = {
    back: getJointStress('back'),
    hip: getJointStress('hip'),
    knee: getJointStress('knee'),
    ankle: getJointStress('ankle'),
  };

  const getColor = (level: 'low' | 'moderate' | 'high', isHovered: boolean) => {
    const colors = {
      low: isHovered ? '#22c55e' : '#4ade80',      // Green
      moderate: isHovered ? '#eab308' : '#facc15', // Yellow
      high: isHovered ? '#dc2626' : '#ef4444',     // Red
    };
    return colors[level];
  };

  const getGlowColor = (level: 'low' | 'moderate' | 'high') => {
    const colors = {
      low: 'rgba(74, 222, 128, 0.4)',
      moderate: 'rgba(250, 204, 21, 0.4)',
      high: 'rgba(239, 68, 68, 0.5)',
    };
    return colors[level];
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-center justify-center">
      {/* Human Body SVG */}
      <div className="relative">
        <svg
          viewBox="0 0 200 400"
          className="w-48 h-96 md:w-56 md:h-[420px]"
          style={{ filter: 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3))' }}
        >
          {/* Body outline - simplified human silhouette */}
          <defs>
            <linearGradient id="bodyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#374151" />
              <stop offset="100%" stopColor="#1f2937" />
            </linearGradient>
            {/* Glow filters for each stress level */}
            <filter id="glowLow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          {/* Head */}
          <ellipse cx="100" cy="35" rx="25" ry="30" fill="url(#bodyGradient)" stroke="#4b5563" strokeWidth="1" />
          
          {/* Neck */}
          <rect x="90" y="60" width="20" height="15" fill="url(#bodyGradient)" />
          
          {/* Torso */}
          <path
            d="M60 75 L140 75 L135 180 L65 180 Z"
            fill="url(#bodyGradient)"
            stroke="#4b5563"
            strokeWidth="1"
          />
          
          {/* Arms */}
          <path d="M60 75 L35 85 L25 145 L35 150 L50 100 L60 90" fill="url(#bodyGradient)" stroke="#4b5563" strokeWidth="1" />
          <path d="M140 75 L165 85 L175 145 L165 150 L150 100 L140 90" fill="url(#bodyGradient)" stroke="#4b5563" strokeWidth="1" />
          
          {/* Legs */}
          <path d="M65 180 L55 300 L50 380 L70 385 L80 300 L85 200" fill="url(#bodyGradient)" stroke="#4b5563" strokeWidth="1" />
          <path d="M135 180 L145 300 L150 380 L130 385 L120 300 L115 200" fill="url(#bodyGradient)" stroke="#4b5563" strokeWidth="1" />
          
          {/* Interactive Joint Markers */}
          
          {/* Back/Spine - upper back area */}
          <ellipse
            cx="100"
            cy="120"
            rx="20"
            ry="25"
            fill={getColor(joints.back.level, selectedJoint === 'back')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            style={{
              filter: selectedJoint === 'back' ? 'url(#glowLow)' : 'none',
              boxShadow: `0 0 20px ${getGlowColor(joints.back.level)}`
            }}
            onMouseEnter={() => setSelectedJoint('back')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'back' ? null : 'back')}
          />
          <text x="100" y="124" textAnchor="middle" fontSize="10" fill="white" fontWeight="bold" pointerEvents="none">
            BACK
          </text>
          
          {/* Hip joints */}
          <ellipse
            cx="80"
            cy="190"
            rx="15"
            ry="12"
            fill={getColor(joints.hip.level, selectedJoint === 'hip')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            onMouseEnter={() => setSelectedJoint('hip')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'hip' ? null : 'hip')}
          />
          <ellipse
            cx="120"
            cy="190"
            rx="15"
            ry="12"
            fill={getColor(joints.hip.level, selectedJoint === 'hip')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            onMouseEnter={() => setSelectedJoint('hip')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'hip' ? null : 'hip')}
          />
          <text x="100" y="194" textAnchor="middle" fontSize="9" fill="white" fontWeight="bold" pointerEvents="none">
            HIP
          </text>
          
          {/* Knee joints */}
          <ellipse
            cx="65"
            cy="285"
            rx="14"
            ry="18"
            fill={getColor(joints.knee.level, selectedJoint === 'knee')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            onMouseEnter={() => setSelectedJoint('knee')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'knee' ? null : 'knee')}
          />
          <ellipse
            cx="135"
            cy="285"
            rx="14"
            ry="18"
            fill={getColor(joints.knee.level, selectedJoint === 'knee')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            onMouseEnter={() => setSelectedJoint('knee')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'knee' ? null : 'knee')}
          />
          <text x="65" y="289" textAnchor="middle" fontSize="8" fill="white" fontWeight="bold" pointerEvents="none">
            KNEE
          </text>
          <text x="135" y="289" textAnchor="middle" fontSize="8" fill="white" fontWeight="bold" pointerEvents="none">
            KNEE
          </text>
          
          {/* Ankle joints */}
          <ellipse
            cx="60"
            cy="365"
            rx="12"
            ry="10"
            fill={getColor(joints.ankle.level, selectedJoint === 'ankle')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            onMouseEnter={() => setSelectedJoint('ankle')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'ankle' ? null : 'ankle')}
          />
          <ellipse
            cx="140"
            cy="365"
            rx="12"
            ry="10"
            fill={getColor(joints.ankle.level, selectedJoint === 'ankle')}
            opacity="0.8"
            className="cursor-pointer transition-all duration-300"
            onMouseEnter={() => setSelectedJoint('ankle')}
            onMouseLeave={() => setSelectedJoint(null)}
            onClick={() => setSelectedJoint(selectedJoint === 'ankle' ? null : 'ankle')}
          />
          <text x="60" y="369" textAnchor="middle" fontSize="7" fill="white" fontWeight="bold" pointerEvents="none">
            ANK
          </text>
          <text x="140" y="369" textAnchor="middle" fontSize="7" fill="white" fontWeight="bold" pointerEvents="none">
            ANK
          </text>
        </svg>
      </div>

      {/* Joint Details Panel */}
      <div className="flex-1 max-w-md space-y-4">
        {/* Legend */}
        <div className="p-4 bg-muted/30 rounded-lg">
          <h4 className="font-semibold mb-3 text-sm">Color Legend</h4>
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span className="text-sm">Low Stress</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-yellow-400"></div>
              <span className="text-sm">Moderate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span className="text-sm">High Stress</span>
            </div>
          </div>
        </div>

        {/* Selected Joint Info or All Joints Summary */}
        {selectedJoint ? (
          <div className={`p-4 rounded-lg border-2 transition-all duration-300 ${
            joints[selectedJoint as keyof typeof joints].level === 'high' 
              ? 'bg-red-100 dark:bg-red-950 border-red-500' 
              : joints[selectedJoint as keyof typeof joints].level === 'moderate'
              ? 'bg-yellow-100 dark:bg-yellow-950 border-yellow-500'
              : 'bg-green-100 dark:bg-green-950 border-green-500'
          }`}>
            <h4 className={`font-bold text-lg capitalize mb-2 ${
              joints[selectedJoint as keyof typeof joints].level === 'high' 
                ? 'text-red-700 dark:text-red-300' 
                : joints[selectedJoint as keyof typeof joints].level === 'moderate'
                ? 'text-yellow-700 dark:text-yellow-300'
                : 'text-green-700 dark:text-green-300'
            }`}>
              {selectedJoint} Joint
            </h4>
            <p className={`text-sm ${
              joints[selectedJoint as keyof typeof joints].level === 'high' 
                ? 'text-red-800 dark:text-red-200' 
                : joints[selectedJoint as keyof typeof joints].level === 'moderate'
                ? 'text-yellow-800 dark:text-yellow-200'
                : 'text-green-800 dark:text-green-200'
            }`}>
              {joints[selectedJoint as keyof typeof joints].description}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground mb-3">
              Click or hover over a joint to see details
            </p>
            {Object.entries(joints).map(([key, joint]) => (
              <div 
                key={key}
                className={`p-3 rounded-lg flex items-center justify-between cursor-pointer transition-all hover:scale-[1.02] ${
                  joint.level === 'high' 
                    ? 'bg-red-100 dark:bg-red-950/50' 
                    : joint.level === 'moderate'
                    ? 'bg-yellow-100 dark:bg-yellow-950/50'
                    : 'bg-green-100 dark:bg-green-950/50'
                }`}
                onClick={() => setSelectedJoint(key)}
              >
                <span className="font-medium capitalize">{key}</span>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-semibold ${
                    joint.level === 'high' 
                      ? 'text-red-600 dark:text-red-400' 
                      : joint.level === 'moderate'
                      ? 'text-yellow-600 dark:text-yellow-400'
                      : 'text-green-600 dark:text-green-400'
                  }`}>
                    {joint.level.toUpperCase()}
                  </span>
                  <div className={`w-3 h-3 rounded-full ${
                    joint.level === 'high' ? 'bg-red-500' : joint.level === 'moderate' ? 'bg-yellow-400' : 'bg-green-500'
                  }`}></div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tip */}
        <div className="p-3 bg-blue-50 dark:bg-blue-950/50 rounded-lg border border-blue-200 dark:border-blue-800">
          <p className="text-xs text-blue-700 dark:text-blue-300">
            <strong>💡 Tip:</strong> Red joints need immediate attention. Adjust your form to reduce stress on these areas and prevent injury.
          </p>
        </div>
      </div>
    </div>
  );
}
