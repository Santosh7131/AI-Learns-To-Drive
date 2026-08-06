import { Component, type ReactNode } from "react";

interface Props {
  fallback: ReactNode;
  children: ReactNode;
}
interface State {
  hasError: boolean;
}

// Catches render/runtime errors in its subtree (e.g. a lost WebGL context or a
// Three.js failure) and shows a fallback instead of blanking the whole app.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    console.error("Scene error (falling back):", error);
  }

  render() {
    return this.state.hasError ? this.props.fallback : this.props.children;
  }
}
