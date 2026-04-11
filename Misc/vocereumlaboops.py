import random
import csv
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comprehensive_sales_data(filename="comprehensive_sales_data.csv", num_records=500):
    """
    Generate realistic sales data with multiple dimensions for analysis
    
    Args:
        filename (str): Output CSV filename
        num_records (int): Number of sales records to generate
    
    Returns:
        str: Confirmation message with file details
    """
    
    # Define product catalog with realistic pricing
    products = {
        'Laptop': {'base_price': 999.99, 'category': 'Electronics', 'cost': 600},
        'Desktop': {'base_price': 1299.99, 'category': 'Electronics', 'cost': 800},
        'Tablet': {'base_price': 499.99, 'category': 'Electronics', 'cost': 300},
        'Smartphone': {'base_price': 799.99, 'category': 'Electronics', 'cost': 400},
        'Monitor': {'base_price': 299.99, 'category': 'Electronics', 'cost': 150},
        'Keyboard': {'base_price': 79.99, 'category': 'Accessories', 'cost': 25},
        'Mouse': {'base_price': 29.99, 'category': 'Accessories', 'cost': 10},
        'Headphones': {'base_price': 199.99, 'category': 'Accessories', 'cost': 80},
        'Webcam': {'base_price': 89.99, 'category': 'Accessories', 'cost': 35},
        'Speaker': {'base_price': 149.99, 'category': 'Accessories', 'cost': 60},
        'Router': {'base_price': 129.99, 'category': 'Networking', 'cost': 50},
        'Switch': {'base_price': 199.99, 'category': 'Networking', 'cost': 80},
        'Cable': {'base_price': 19.99, 'category': 'Accessories', 'cost': 5}
    }
    
    # Salesperson profiles with different performance characteristics
    salespeople = {
        'Alice Johnson': {'skill_level': 0.9, 'region': 'North', 'start_date': '2020-01-15'},
        'Bob Smith': {'skill_level': 0.7, 'region': 'South', 'start_date': '2021-06-10'},
        'Charlie Brown': {'skill_level': 0.8, 'region': 'East', 'start_date': '2019-03-22'},
        'Diana Ross': {'skill_level': 0.85, 'region': 'West', 'start_date': '2020-09-05'},
        'Eve Williams': {'skill_level': 0.75, 'region': 'North', 'start_date': '2022-02-14'},
        'Frank Miller': {'skill_level': 0.65, 'region': 'South', 'start_date': '2021-11-30'},
        'Grace Lee': {'skill_level': 0.88, 'region': 'East', 'start_date': '2020-07-18'},
        'Henry Davis': {'skill_level': 0.72, 'region': 'West', 'start_date': '2021-04-12'}
    }
    
    # Regional characteristics
    regions = {
        'North': {'market_strength': 1.1, 'competition': 0.8},
        'South': {'market_strength': 0.9, 'competition': 1.2},
        'East': {'market_strength': 1.05, 'competition': 1.0},
        'West': {'market_strength': 1.15, 'competition': 0.9}
    }
    
    # Customer segments
    customer_types = {
        'Enterprise': {'volume_multiplier': 3.0, 'discount_rate': 0.15, 'frequency': 0.2},
        'SMB': {'volume_multiplier': 1.5, 'discount_rate': 0.08, 'frequency': 0.4},
        'Individual': {'volume_multiplier': 1.0, 'discount_rate': 0.02, 'frequency': 0.4}
    }
    
    # Generate sales records
    sales_records = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(num_records):
        # Random date within the year
        random_days = random.randint(0, (end_date - start_date).days)
        sale_date = start_date + timedelta(days=random_days)
        
        # Select random elements
        product_name = random.choice(list(products.keys()))
        product_info = products[product_name]
        salesperson_name = random.choice(list(salespeople.keys()))
        salesperson_info = salespeople[salesperson_name]
        region = salesperson_info['region']
        customer_type = random.choices(
            list(customer_types.keys()),
            weights=[ct['frequency'] for ct in customer_types.values()]
        )[0]
        
        # Calculate quantity based on product type and customer
        base_quantity = 1
        if product_info['category'] == 'Accessories':
            base_quantity = random.randint(1, 5)
        elif product_info['category'] == 'Electronics':
            base_quantity = random.randint(1, 3)
        
        quantity = int(base_quantity * customer_types[customer_type]['volume_multiplier'])
        quantity = max(1, quantity)  # Ensure at least 1
        
        # Calculate price with variations
        base_price = product_info['base_price']
        
        # Apply regional market strength
        price_modifier = regions[region]['market_strength']
        
        # Apply salesperson skill (better salespeople get better prices)
        skill_modifier = 0.95 + (salesperson_info['skill_level'] * 0.1)
        
        # Apply customer discount
        discount = customer_types[customer_type]['discount_rate']
        
        # Add seasonal variation
        month = sale_date.month
        seasonal_modifier = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_modifier = 1.15
        elif month in [6, 7, 8]:  # Summer
            seasonal_modifier = 0.95
        
        # Calculate final price
        final_price = base_price * price_modifier * skill_modifier * seasonal_modifier * (1 - discount)
        final_price = round(final_price, 2)
        
        # Generate customer ID
        customer_id = f"CUST{random.randint(10000, 99999)}"
        
        # Create record
        record = {
            'Date': sale_date.strftime('%Y-%m-%d'),
            'Product': product_name,
            'Category': product_info['category'],
            'Quantity': quantity,
            'Unit_Price': final_price,
            'Total_Revenue': round(quantity * final_price, 2),
            'Cost': round(quantity * product_info['cost'], 2),
            'Profit': round((quantity * final_price) - (quantity * product_info['cost']), 2),
            'Salesperson': salesperson_name,
            'Region': region,
            'Customer_Type': customer_type,
            'Customer_ID': customer_id,
            'Discount_Applied': round(discount * 100, 1),
            'Quarter': f"Q{(sale_date.month - 1) // 3 + 1}",
            'Month': sale_date.strftime('%B'),
            'Day_of_Week': sale_date.strftime('%A')
        }
        
        sales_records.append(record)
    
    # Sort by date
    sales_records.sort(key=lambda x: x['Date'])
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = sales_records[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sales_records)
    
    # Calculate summary statistics
    total_revenue = sum(record['Total_Revenue'] for record in sales_records)
    total_profit = sum(record['Profit'] for record in sales_records)
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
    
    return f"""
Generated {filename} with {num_records} sales records
Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Total Revenue: ${total_revenue:,.2f}
Total Profit: ${total_profit:,.2f}
Profit Margin: {profit_margin:.1f}%
Products: {len(products)} different products
Salespeople: {len(salespeople)} sales representatives
Regions: {len(regions)} regions
"""

def process_sales_data(filename):
    """
    Process sales CSV and generate comprehensive business insights
    
    Args:
        filename (str): Path to CSV file
    
    Returns:
        dict: Comprehensive analysis results
    """
    try:
        # Read CSV data
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        if len(lines) < 2:
            return {'error': 'File is empty or has no data rows'}
        
        # Skip header
        header = lines[0].strip().split(',')
        data_lines = lines[1:]
        
        # Initialize aggregation dictionaries
        product_revenue = {}
        product_profit = {}
        product_quantity = {}
        salesperson_revenue = {}
        salesperson_profit = {}
        salesperson_sales_count = {}
        region_revenue = {}
        region_profit = {}
        customer_type_revenue = {}
        category_revenue = {}
        monthly_revenue = {}
        quarterly_revenue = {}
        
        total_revenue = 0
        total_profit = 0
        total_quantity = 0
        
        # Process each line
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 15:  # Ensure we have all expected columns
                try:
                    # Extract data (based on our generated CSV structure)
                    date = parts[0]
                    product = parts[1]
                    category = parts[2]
                    quantity = int(parts[3])
                    unit_price = float(parts[4])
                    revenue = float(parts[5])
                    cost = float(parts[6])
                    profit = float(parts[7])
                    salesperson = parts[8]
                    region = parts[9]
                    customer_type = parts[10]
                    customer_id = parts[11]
                    discount = float(parts[12])
                    quarter = parts[13]
                    month = parts[14]
                    day_of_week = parts[15] if len(parts) > 15 else 'Unknown'
                    
                    # Aggregate totals
                    total_revenue += revenue
                    total_profit += profit
                    total_quantity += quantity
                    
                    # Aggregate by product
                    product_revenue[product] = product_revenue.get(product, 0) + revenue
                    product_profit[product] = product_profit.get(product, 0) + profit
                    product_quantity[product] = product_quantity.get(product, 0) + quantity
                    
                    # Aggregate by salesperson
                    salesperson_revenue[salesperson] = salesperson_revenue.get(salesperson, 0) + revenue
                    salesperson_profit[salesperson] = salesperson_profit.get(salesperson, 0) + profit
                    salesperson_sales_count[salesperson] = salesperson_sales_count.get(salesperson, 0) + 1
                    
                    # Aggregate by region
                    region_revenue[region] = region_revenue.get(region, 0) + revenue
                    region_profit[region] = region_profit.get(region, 0) + profit
                    
                    # Aggregate by customer type
                    customer_type_revenue[customer_type] = customer_type_revenue.get(customer_type, 0) + revenue
                    
                    # Aggregate by category
                    category_revenue[category] = category_revenue.get(category, 0) + revenue
                    
                    # Aggregate by time periods
                    monthly_revenue[month] = monthly_revenue.get(month, 0) + revenue
                    quarterly_revenue[quarter] = quarterly_revenue.get(quarter, 0) + revenue
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line: {line.strip()}")
                    continue
        
        # Calculate additional metrics
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        avg_order_value = total_revenue / len(data_lines) if data_lines else 0
        
        # Find top performers
        top_product = max(product_revenue.items(), key=lambda x: x[1]) if product_revenue else ("None", 0)
        top_salesperson = max(salesperson_revenue.items(), key=lambda x: x[1]) if salesperson_revenue else ("None", 0)
        top_region = max(region_revenue.items(), key=lambda x: x[1]) if region_revenue else ("None", 0)
        
        # Calculate salesperson efficiency (revenue per sale)
        salesperson_efficiency = {}
        for sp in salesperson_revenue:
            if salesperson_sales_count[sp] > 0:
                salesperson_efficiency[sp] = salesperson_revenue[sp] / salesperson_sales_count[sp]
        
        return {
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'total_quantity': total_quantity,
            'profit_margin': profit_margin,
            'avg_order_value': avg_order_value,
            'total_transactions': len(data_lines),
            
            # Product insights
            'product_revenue': product_revenue,
            'product_profit': product_profit,
            'product_quantity': product_quantity,
            'top_product': top_product,
            
            # Salesperson insights
            'salesperson_revenue': salesperson_revenue,
            'salesperson_profit': salesperson_profit,
            'salesperson_efficiency': salesperson_efficiency,
            'top_salesperson': top_salesperson,
            
            # Regional insights
            'region_revenue': region_revenue,
            'region_profit': region_profit,
            'top_region': top_region,
            
            # Other dimensions
            'customer_type_revenue': customer_type_revenue,
            'category_revenue': category_revenue,
            'monthly_revenue': monthly_revenue,
            'quarterly_revenue': quarterly_revenue
        }
        
    except FileNotFoundError:
        return {'error': f'File {filename} not found'}
    except Exception as e:
        return {'error': f'Error processing file: {str(e)}'}

def display_comprehensive_analysis(analysis_results):
    """
    Display comprehensive analysis results in a formatted way
    
    Args:
        analysis_results (dict): Results from process_sales_data
    """
    if 'error' in analysis_results:
        print(f"Error: {analysis_results['error']}")
        return
    
    print("=" * 80)
    print("COMPREHENSIVE SALES ANALYSIS REPORT")
    print("=" * 80)
    
    # Overall Performance
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"{'─' * 40}")
    print(f"Total Revenue:        ${analysis_results['total_revenue']:,.2f}")
    print(f"Total Profit:         ${analysis_results['total_profit']:,.2f}")
    print(f"Profit Margin:        {analysis_results['profit_margin']:.1f}%")
    print(f"Total Transactions:   {analysis_results['total_transactions']:,}")
    print(f"Average Order Value:  ${analysis_results['avg_order_value']:,.2f}")
    print(f"Total Units Sold:     {analysis_results['total_quantity']:,}")
    
    # Top Performers
    print(f"\n🏆 TOP PERFORMERS")
    print(f"{'─' * 40}")
    print(f"Best Product:         {analysis_results['top_product'][0]} (${analysis_results['top_product'][1]:,.2f})")
    print(f"Best Salesperson:     {analysis_results['top_salesperson'][0]} (${analysis_results['top_salesperson'][1]:,.2f})")
    print(f"Best Region:          {analysis_results['top_region'][0]} (${analysis_results['top_region'][1]:,.2f})")
    
    # Product Analysis
    print(f"\n🛍️ PRODUCT ANALYSIS")
    print(f"{'─' * 40}")
    sorted_products = sorted(analysis_results['product_revenue'].items(), 
                           key=lambda x: x[1], reverse=True)
    print("Top 5 Products by Revenue:")
    for i, (product, revenue) in enumerate(sorted_products[:5], 1):
        profit = analysis_results['product_profit'].get(product, 0)
        quantity = analysis_results['product_quantity'].get(product, 0)
        margin = (profit / revenue * 100) if revenue > 0 else 0
        print(f"  {i}. {product:<15} Revenue: ${revenue:>8,.0f} | "
              f"Profit: ${profit:>6,.0f} | Margin: {margin:>4.1f}% | Units: {quantity:>3}")
    
    # Salesperson Analysis
    print(f"\n👥 SALESPERSON ANALYSIS")
    print(f"{'─' * 40}")
    sorted_salespeople = sorted(analysis_results['salesperson_revenue'].items(), 
                              key=lambda x: x[1], reverse=True)
    print("Salesperson Performance:")
    for name, revenue in sorted_salespeople:
        profit = analysis_results['salesperson_profit'].get(name, 0)
        efficiency = analysis_results['salesperson_efficiency'].get(name, 0)
        print(f"  {name:<15} Revenue: ${revenue:>8,.0f} | "
              f"Profit: ${profit:>6,.0f} | Avg/Sale: ${efficiency:>6,.0f}")
    
    # Regional Analysis
    print(f"\n🌍 REGIONAL ANALYSIS")
    print(f"{'─' * 40}")
    for region, revenue in sorted(analysis_results['region_revenue'].items(), 
                                key=lambda x: x[1], reverse=True):
        profit = analysis_results['region_profit'].get(region, 0)
        margin = (profit / revenue * 100) if revenue > 0 else 0
        print(f"  {region:<10} Revenue: ${revenue:>8,.0f} | "
              f"Profit: ${profit:>6,.0f} | Margin: {margin:>4.1f}%")
    
    # Customer Type Analysis
    print(f"\n🎯 CUSTOMER SEGMENT ANALYSIS")
    print(f"{'─' * 40}")
    for customer_type, revenue in sorted(analysis_results['customer_type_revenue'].items(), 
                                       key=lambda x: x[1], reverse=True):
        percentage = (revenue / analysis_results['total_revenue'] * 100) if analysis_results['total_revenue'] > 0 else 0
        print(f"  {customer_type:<12} ${revenue:>8,.0f} ({percentage:>4.1f}%)")
    
    # Category Analysis
    print(f"\n📂 CATEGORY ANALYSIS")
    print(f"{'─' * 40}")
    for category, revenue in sorted(analysis_results['category_revenue'].items(), 
                                  key=lambda x: x[1], reverse=True):
        percentage = (revenue / analysis_results['total_revenue'] * 100) if analysis_results['total_revenue'] > 0 else 0
        print(f"  {category:<15} ${revenue:>8,.0f} ({percentage:>4.1f}%)")
    
    # Quarterly Performance
    print(f"\n📅 QUARTERLY PERFORMANCE")
    print(f"{'─' * 40}")
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    for quarter in quarters:
        revenue = analysis_results['quarterly_revenue'].get(quarter, 0)
        print(f"  {quarter}: ${revenue:>10,.0f}")

def create_visualizations(analysis_results, save_plots=True):
    """
    Create comprehensive visualizations of the sales data
    
    Args:
        analysis_results (dict): Results from process_sales_data
        save_plots (bool): Whether to save plots to files
    """
    if 'error' in analysis_results:
        print(f"Cannot create visualizations: {analysis_results['error']}")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sales Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Top Products by Revenue
    top_products = sorted(analysis_results['product_revenue'].items(), 
                         key=lambda x: x[1], reverse=True)[:8]
    products, revenues = zip(*top_products)
    
    axes[0, 0].bar(range(len(products)), revenues, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('Top Products by Revenue')
    axes[0, 0].set_xlabel('Products')
    axes[0, 0].set_ylabel('Revenue ($)')
    axes[0, 0].set_xticks(range(len(products)))
    axes[0, 0].set_xticklabels(products, rotation=45, ha='right')
    
    # Format y-axis to show values in thousands
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 2. Salesperson Performance
    sp_data = sorted(analysis_results['salesperson_revenue'].items(), 
                    key=lambda x: x[1], reverse=True)
    names, sp_revenues = zip(*sp_data)
    
    axes[0, 1].barh(names, sp_revenues, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_title('Salesperson Revenue Performance')
    axes[0, 1].set_xlabel('Revenue ($)')
    axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 3. Regional Distribution (Pie Chart)
    regions = list(analysis_results['region_revenue'].keys())
    region_values = list(analysis_results['region_revenue'].values())
    
    axes[0, 2].pie(region_values, labels=regions, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Revenue Distribution by Region')
    
    # 4. Customer Type Analysis
    customer_types = list(analysis_results['customer_type_revenue'].keys())
    customer_revenues = list(analysis_results['customer_type_revenue'].values())
    
    axes[1, 0].bar(customer_types, customer_revenues, color='coral', edgecolor='darkred')
    axes[1, 0].set_title('Revenue by Customer Type')
    axes[1, 0].set_ylabel('Revenue ($)')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 5. Quarterly Trends
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    quarterly_values = [analysis_results['quarterly_revenue'].get(q, 0) for q in quarters]
    
    axes[1, 1].plot(quarters, quarterly_values, marker='o', linewidth=3, markersize=8, color='purple')
    axes[1, 1].set_title('Quarterly Revenue Trend')
    axes[1, 1].set_ylabel('Revenue ($)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 6. Category Performance
    categories = list(analysis_results['category_revenue'].keys())
    category_values = list(analysis_results['category_revenue'].values())
    
    axes[1, 2].bar(categories, category_values, color='gold', edgecolor='orange')
    axes[1, 2].set_title('Revenue by Product Category')
    axes[1, 2].set_ylabel('Revenue ($)')
    axes[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('sales_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("📊 Saved visualization dashboard as 'sales_analysis_dashboard.png'")
    
    plt.show()

def main():
    """
    Main function to demonstrate the complete sales analysis system
    """
    print("🚀 Starting Comprehensive Sales Data Analysis System")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n📝 Step 1: Generating comprehensive sales data...")
    generation_result = generate_comprehensive_sales_data("advanced_sales_data.csv", 1000)
    print(generation_result)
    
    # Step 2: Process the data
    print("\n🔍 Step 2: Processing sales data...")
    analysis_results = process_sales_data("advanced_sales_data.csv")
    
    # Step 3: Display comprehensive analysis
    print("\n📋 Step 3: Displaying analysis results...")
    display_comprehensive_analysis(analysis_results)
    
    # Step 4: Create visualizations
    print("\n📊 Step 4: Creating visualizations...")
    create_visualizations(analysis_results)
    
    print("\n✅ Analysis complete! Check the generated files and visualizations.")
    
    return analysis_results

if __name__ == "__main__":
    results = main()