"""Command-line interface for reasoning evaluator."""

import argparse
import os
from reasoning_evaluator import FullEvaluator, ChainComparator, load_dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning question chains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a dataset
  reasoning-eval evaluate dataset.json --output results/
  
  # Compare good vs weak chains
  reasoning-eval compare dataset.json --output comparisons/
  
  # Evaluate without semantic model (faster, less accurate)
  reasoning-eval evaluate dataset.json --no-semantic
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate reasoning chains')
    eval_parser.add_argument('dataset', help='Path to dataset JSON file')
    eval_parser.add_argument('--output', '-o', default='evaluation_results',
                            help='Output directory (default: evaluation_results)')
    eval_parser.add_argument('--no-semantic', action='store_true',
                            help='Disable semantic model (faster but less accurate)')
    
    # Compare command
    comp_parser = subparsers.add_parser('compare', help='Compare good vs weak chains')
    comp_parser.add_argument('dataset', help='Path to dataset JSON file')
    comp_parser.add_argument('--output', '-o', default='comparisons',
                            help='Output directory (default: comparisons)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    good_chains, weak_chains = load_dataset(args.dataset)
    print(f"✓ Loaded {len(good_chains)} good chains and {len(weak_chains)} weak chains")
    
    if args.command == 'evaluate':
        # Create evaluator
        use_semantic = not args.no_semantic
        evaluator = FullEvaluator(use_semantic_model=use_semantic)
        
        # Evaluate all chains
        print(f"\nEvaluating chains...")
        os.makedirs(args.output, exist_ok=True)
        
        for chain in good_chains + weak_chains:
            report = evaluator.evaluate_chain(chain, save_dir=args.output)
        
        print(f"\n✅ Evaluation complete! Results saved to {args.output}/")
    
    elif args.command == 'compare':
        # Compare chains
        print(f"\nComparing good vs weak chains...")
        ChainComparator.compare_chains(good_chains, weak_chains, save_dir=args.output)
        print(f"\n✅ Comparison complete! Results saved to {args.output}/")


if __name__ == '__main__':
    main()
