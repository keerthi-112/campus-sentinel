import { Component } from "react";

// Keeps one broken page from blanking the whole dashboard.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidUpdate(prev) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }
  render() {
    if (this.state.error)
      return (
        <div className="card">
          <h3>This page hit an error</h3>
          <div className="alert">{String(this.state.error.message || this.state.error)}</div>
          <button onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      );
    return this.props.children;
  }
}
