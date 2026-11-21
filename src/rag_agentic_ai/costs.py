"""Paid-tier prices, so you can see what a call would on a paid plan."""


PRICES: dict[ str, tuple[ float, float ]  ] = {
     "gemini-3.5-flash" : ( 1.50, 9.00 ),
}

def cost_usd( input_tokens: int, output_tokens: int, price_in:float , price_out:float ) -> float:
    """Return the cost in USD for a call with the given number of input and output tokens."""
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out