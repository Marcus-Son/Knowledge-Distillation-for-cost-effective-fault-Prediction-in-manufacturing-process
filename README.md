# Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process 

## Background
In manufacturing, it is important to predict the condition of a product in advance to reduce repair and replacement costs. To do this, basic and advanced inspections are typically performed on products. Basic inspections are common for all products, while advanced inspections are performed for some products at an additional cost.
- Basic Inspection
  Purpose: To verify the general quality and performance of the product.
  Scope: Performed on all products. It is considered a standard part of the manufacturing process.
  Features: Includes basic quality inspection items and verifies the basic safety and functionality of the product. For example, it might include visual inspections, basic functional tests, and simple physical tests.
  Cost: Relatively inexpensive. In most manufacturing processes, basic inspections are considered standard procedure and are performed at no additional cost.
- Advanced Inspection
  Purpose: To verify the detailed quality and performance of a product in greater depth.
  Scope: Performed selectively, advanced inspections typically entail additional costs.
  Features: Includes more sophisticated and systematic testing. For example, advanced materials testing, detailed performance evaluations, and complex safety testing. These inspections are designed to uncover deeper flaws in a particular product.
  Cost: They are more expensive than basic inspections and sometimes require specialized equipment or specialized expertise, which can make it cost ineffective to perform advanced inspections on every product.

![image](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/8ec0f1ee-8902-45b8-8e3e-dbf302bc5b60)

While these inspections are important for ensuring product quality and identifying defects in advance, they also present a challenge to manage costs and time efficiently. Basic inspections are essential for all products, but due to their limited scope, they may not detect some defects. Advanced inspections, on the other hand, provide more accurate and detailed information, but can be costly to apply to all products. Balancing these two inspection methods therefore becomes a major challenge in manufacturing.

## Problems
- Inspection cost and time: Performing extensive quality inspections on every product is inefficient in terms of cost and time. Advanced inspections, in particular, require additional costs and resources, which increase overall manufacturing costs.
- Accuracy of defect prediction: Due to limited data and inspection items, defect prediction can be inaccurate. This impacts product quality control and increases the risk of defective products reaching the market.
- Limitations of advanced inspections: Performing advanced inspections on every product is cost impractical. However, some defects may not be detected by basic inspections alone.
- Difficulty in making cost-effective decisions: Deciding which products should be subjected to advanced inspection is a complex task, made more difficult without proper data analysis and predictive models.

## Proposed Method
To solve the above problems, we leverage Knowledge Distillation. This is a technique that improves the performance of the base model by transferring knowledge from the advanced model to the base model. In the training phase of the prediction model, it assumes that all features are available, and predicts the status of the product with the basic model for products where only basic inspection has been performed, and with the advanced model for products where both basic and advanced inspection have been performed. In this study, we assume that the advanced model has better prediction performance than the basic model.

![image](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/46818e9f-d5bf-4f6f-9c9e-51091dd778ea)


## Goals
- Developing a better-performing base model: Leveraging Knowledge Distillation to transfer knowledge from advanced models to the base model, thereby improving the performance of the base model. The main goal is to increase the prediction accuracy for products with only basic inspections.
- Cost-Effective Defect Prediction: For defect prediction in the manufacturing process, the goal is to develop cost-effective models that minimize costly advanced inspections while maintaining high prediction accuracy.
- Optimize advanced inspection sampling: Instead of performing advanced inspections on all products, find ways to perform advanced inspections on only some products using efficient sampling methods and use this data to improve defect prediction for all products.

## Data description
1. Data description
   - In the semiconductor manufacturing process, each chip on a wafer is inspected as the wafer goes through the manufacturing process. As the wafer goes through the manufacturing process.
   - This data can be used to predict in advance whether a wafer will be defective.
     ![image](https://github.com/ShawnSon-hub/Knowledge-Distillation-for-cost-effective-fault-Prediction-in-manufacturing-process/assets/124177883/78f9db95-a31f-46fe-a0bb-6c7bff16b1af)

3. Data Preprocessing

## 


