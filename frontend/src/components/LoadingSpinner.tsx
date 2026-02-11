import { Activity } from "lucide-react";

export function LoadingSpinner({ message = "Loading..." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-primary"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <Activity className="h-6 w-6 text-primary animate-pulse" />
        </div>
      </div>
      <p className="mt-4 text-sm text-muted-foreground animate-pulse">{message}</p>
    </div>
  );
}
