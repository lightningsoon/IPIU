import tensorflow as tf
import eval
import PSP_model

SIZE = 200  # 图像尺寸
batch_size = 16
STEPS = 100
dataset_size = 1111
save_dir = './'  # 保存网络路径
global_step = tf.Variable(0, trainable=False)
log_dir = './log/'
X = tf.placeholder(tf.float32, shape=(None, SIZE, SIZE, 3), name="input_x")
Y = tf.placeholder(tf.float32, shape=(None, SIZE, SIZE, 1), name="input_y")
y_ = PSP_model.y  # 最终输出结果，列表

learing_rate = tf.train.exponential_decay(0.1, global_step, STEPS // 50, 0.9, staircase=True)
tf.summary.scalar('learning_rate',learing_rate)
loss = eval.accuracy(Y, y_)  # 损失应该加入L2正则化
tf.summary.scalar('loss',loss)
train_step = tf.train.AdamOptimizer(learing_rate).minimize(loss, global_step=global_step)  # 可以用滑动平均模型改进
saver = tf.train.Saver()
merged=tf.summary.merge_all()
with tf.Session as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        _, _ = sess.run(
            [train_step,global_step],
            feed_dict={x: X[start:end], y: Y[start:end]})
        if i%20 ==0:
            _, loss_value, step_value,summary = sess.run(
                [train_step, loss, global_step,merged],
                feed_dict={x: X[start:end], y: Y[start:end]})
            print("Loss value at %d step:%f" % (step_value,loss_value))
            saver.save(sess, save_dir)
            writer.add_summary(merged,global_step=global_step)
