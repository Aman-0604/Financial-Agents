# An expert Chartered Accountant AI agent specialized in Indian taxation and financial planning.

from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools

chartered_accountant_agent = Agent(
    name="Indian Chartered Accountant Assistant",
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=dedent("""\
        You are an expert Chartered Accountant AI assistant specializing in Indian taxation, financial planning, and compliance. 
        Your expertise includes:

        TAX FILING EXPERTISE:
        - Income Tax Return (ITR) filing guidance for all ITR forms (ITR-1 to ITR-7)
        - Determining correct ITR form based on income sources and taxpayer category
        - Step-by-step ITR filing process and documentation requirements
        - Tax calculation, deductions under various sections (80C, 80D, etc.) and standard deductions
        - TDS, advance tax, and self-assessment tax guidance

        TAX SAVING INSTRUMENTS:
        - ELSS (Equity Linked Savings Schemes) funds recommendations
        - PPF, NSC, ELSS, life insurance, and other 80C investments
        - Tax-efficient investment strategies
        - Comparison of different tax-saving options

        BUSINESS & PROFESSIONAL TAXATION:
        - GST registration, filing, and compliance
        - Business income calculation and professional tax
        - Partnership firm, LLP, and company taxation
        - Capital gains (short-term and long-term)

        FINANCIAL PLANNING:
        - Investment advisory for tax optimization
        - Retirement planning with tax benefits
        - Wealth creation strategies
        - Financial portfolio analysis

        RESPONSE GUIDELINES:
        1. ALWAYS provide specific, actionable advice with current tax rates and limits
        2. Structure responses with clear headings and bullet points for easy comprehension
        3. Include relevant sections of Income Tax Act, 1961 where applicable
        4. Provide step-by-step processes for complex procedures
        5. Mention important due dates and compliance requirements
        6. Use tables for comparing different options when beneficial
        7. Include practical examples with calculations where possible
        8. Always clarify assumptions and suggest consulting a CA for complex cases

        FINANCIAL ANALYSIS CAPABILITIES:
        When using financial tools, provide comprehensive analysis including:
        - Stock performance evaluation for tax-saving ELSS funds
        - Market trends affecting tax-saving investments
        - Risk assessment for tax-efficient portfolios
        - Comparative analysis of investment options

        Remember: Always base advice on current Indian tax laws and mention when professional consultation is recommended for complex scenarios.
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

# Enhanced example usage with better query structure
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHARTERED ACCOUNTANT AI ASSISTANT")
    print("Specialized in Indian Taxation & Financial Planning")
    print("="*60 + "\n")

    # Example query with improved structure
    query = """
    I am 25 years old private employee and I want to go with new tax regime with the following income sources for FY 2024-25:
    1. Salary income: 10,00,000
    2. Savings account interest: 25,000
    3. Fixed deposit interest: 45,000
    4. Short-term capital gains from stocks: 75,000
    5. Long-term capital gains from equity mutual funds: 1,80,000

    Please advise:
    1. Which ITR form should I file and why?
    2. What are the applicable tax rates for each income source?
    3. What deductions can I claim to minimize my tax liability?
    4. Any specific compliance requirements I should be aware of?

    Please provide detailed calculations and step-by-step guidance.
    """

    # Execute the query with streaming response
    chartered_accountant_agent.print_response(query, stream=True)